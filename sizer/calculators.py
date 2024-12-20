import scipy.optimize
import scipy.interpolate
import numpy as np

import functools

from math import ceil, copysign, floor


class CalculationError(Exception):
    pass

def conditionFirstOccurrenceIndex(sequence: np.ndarray, condition: np.ndarray) -> int:
    """Return the smallest index of all the elements in `sequence` where `condition` is true.
    """
    try:
        return np.min(np.where(condition))
    except:
        raise CalculationError("condition is never met in this sequence.")

def bandwidth(frequenciesInHertz, frequencyResponse, initialGuess=1e+3):
    """Calculate the frequency at which the absolute value of frequency response drops to 1 / sqrt(2) of its value at 1 Hz.

    Attributes
    ----------

    frequenciesInHertz : 1-D ndarray
        Frequency points in Hz
    frequencyResponse : 1-D ndarray
        Frequency response points, given as an array of complex numbers
    initialGuess : float or int
        Initial guess `x0` for the root finder. Providing reasonable and highly likely initial guess can speed up root finding.

    Frequency response is first interpolated with linear B-spline and then sent to a root finder.
    """
    amplitudeResponse = np.abs(frequencyResponse)
    # amplitudeResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz, amplitudeResponse, bounds_error=False) # interpolate amplitude response with linear b-spline
    # amplitudeAt1Hz = amplitudeResponseInterpolated(1) # get amplitude response at 1 Hz # 38 us
    amplitudeAt1Hz = np.interp(1, frequenciesInHertz, amplitudeResponse, left=np.nan, right=np.nan) # 6 us
    amplitudeAtBandwidth = amplitudeAt1Hz / np.sqrt(2)
    # todo
    try:
        firstOutsideBandwidthFrequency = np.min(np.where(amplitudeResponse < amplitudeAtBandwidth))
        # amplitudeResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz[firstOutsideBandwidthFrequency - 1: firstOutsideBandwidthFrequency + 1], amplitudeResponse[firstOutsideBandwidthFrequency - 1: firstOutsideBandwidthFrequency + 1], bounds_error=False) # interpolate amplitude response with linear b-spline
    # if np.any(amplitudeResponse <= amplitudeAt1Hz / np.sqrt(2)): # check if there exists a point below -3dB
        # return scipy.optimize.root(lambda x: amplitudeResponseInterpolated(x) - amplitudeAtBandwidth, frequenciesInHertz[firstOutsideBandwidthFrequency - 1]).x[0]
        slicedFrequencies = frequenciesInHertz[firstOutsideBandwidthFrequency - 1: firstOutsideBandwidthFrequency + 1]
        slicedAmplitudeResponse = amplitudeResponse[firstOutsideBandwidthFrequency - 1: firstOutsideBandwidthFrequency + 1]
        return scipy.optimize.root(
            lambda x: np.interp(
                x,
                slicedFrequencies,
                slicedAmplitudeResponse,
                left=np.nan,
                right=np.nan
            ) - amplitudeAtBandwidth,
            frequenciesInHertz[firstOutsideBandwidthFrequency - 1]
        ).x[0]
    # else: # if there is no amplitude below -3dB, then no need to compute
    except:
        raise CalculationError("impossible to calculate bandwidth, because the data contains no amplitude point that is below 1 / sqrt(2) times the amplitude at 1 Hz. Try simulating with wider frequency range, or this circuit does not have a bandwidth at all. Amplitude at 1 Hz is {}. Amplitude at {} Hz is {}".format(amplitudeAt1Hz, frequenciesInHertz[-1], amplitudeResponse[-1]))

def unityGainFrequency(frequenciesInHertz, frequencyResponse, initialGuess=1e+3): # 1 ms, special thanks to @HereDrlv
    """Calculate the frequency at which the absolute value of frequency response drops to 1.

    Attributes
    ----------

    frequenciesInHertz : 1-D ndarray
        Frequency points in Hz
    frequencyResponse : 1-D ndarray
        Frequency response points, given as an array of complex numbers
    initialGuess : float or int
        Initial guess `x0` for the root finder. Providing reasonable and highly likely initial guess can speed up root finding.

    Frequency response is first interpolated with linear B-spline and then sent to a root finder.

    Notes
    -----

    SPICE code:

        let cur = 0
        cursor cur right gainvec 1
        let fT = real(frequency[%cur])
    """
    amplitudeResponse = np.abs(frequencyResponse)
    try:
        # closest amplitude to 1.
        i = np.argmin(abs(amplitudeResponse - 1))
        i_max = np.argmax(amplitudeResponse)
        a_max = amplitudeResponse[i_max]
        i_min = np.argmin(amplitudeResponse)
        a_min = amplitudeResponse[i_min]

        # circuit does not reach unity gain at all.
        if a_max < 1:
            i, name = i_max, "max"
        # unity gain is possibly out of bounds.
        elif a_min > 1:
            i, name = i_min, "min"
        # a_min <= 1 and a_max >= 1, should not raise error.
        else:
            # TODO: investigate.
            # print("frequencies:", frequenciesInHertz)
            # print("amplitudes:", amplitudeResponse)

            name = "argmin"

        # Otherwise the circuit does not reach unity gain.
        assert a_min <= 1 and a_max >= 1
    except:
        a_i = amplitudeResponse[i]

        print(f"closest amplitude ({name}): {a_i}", end=", ")
        print(f"frequency: {frequenciesInHertz[i]}")

        # TODO: try out narrower ranges
        # (this one: (0.5, 1.5))
        if round(a_i) == 1:
            return frequenciesInHertz[i]
        else:
            return 0

    # real-valued cursor with amplitude 1.
    i = floatCursorRight(amplitudeResponse, 1)

    f_i = floatCursorGet(frequenciesInHertz, i)

    return f_i

def positiveFeedbackFrequency(frequenciesInHertz, frequencyResponse, initialGuess=1e+3):
    """Calculate the frequency in Hertz at which the phase drops to -180deg.

    Attributes
    ----------

    frequenciesInHertz : 1-D ndarray
        Frequency points in Hz
    frequencyResponse : 1-D ndarray
        Frequency response points, given as an array of complex numbers
    initialGuess : float or int
        Initial guess `x0` for the root finder. Providing reasonable and highly likely initial guess can speed up root finding.
    """
    phaseResponse = np.angle(frequencyResponse, deg=True)
    phaseResponse[np.where(phaseResponse > 0)] -= 360
    try:
        firstBelowNegative180degIndex = np.min(np.where(phaseResponse < -180))
        # phaseResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz[firstBelowNegative180degIndex - 1: firstBelowNegative180degIndex + 1], phaseResponse[firstBelowNegative180degIndex - 1: firstBelowNegative180degIndex + 1], bounds_error=False)
        return scipy.optimize.root(lambda x: np.interp(x, \
        frequenciesInHertz[firstBelowNegative180degIndex - 1: firstBelowNegative180degIndex + 1], \
        phaseResponse[firstBelowNegative180degIndex - 1: firstBelowNegative180degIndex + 1]) + 180, \
        frequenciesInHertz[firstBelowNegative180degIndex - 1]).x[0]
    except:
        raise CalculationError("impossible to calculate the frequency at which phase drops to -180deg, either because the circuit does not reach -180deg at all, or because simulation frequency range is not wide enough.")

def phaseMargin(frequenciesInHertz, frequencyResponse):
    """Calculate the phase margin in degree.

    Attributes
    ----------

    frequenciesInHertz : 1-D ndarray
        Frequency points in Hz
    frequencyResponse : 1-D ndarray
        Frequency response points, given as an array of complex numbers

    Frequency response is first sent to `unityGainFrequency()` to calculate the unity gain frequency, and then frequency response is interpolated with linear B-spline and substituted with unity gain frequency.
    """
    ugf = unityGainFrequency(frequenciesInHertz, frequencyResponse)
    phaseResponse = np.angle(frequencyResponse, deg=True)
    # Note that `np.angle()` returns angles in (-180deg, 180deg], so any phase response that are below -180deg will be returned as if added 360deg, leaving a gap. However, in real practice, phases within (-180deg, -360deg) is drawn below not above to avoid the gap.
    # Attempt to fix this with naive approach.
    phaseResponse[np.where(phaseResponse > 0)] -= 360
    if np.any(phaseResponse <= -180):
        # phaseResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz, phaseResponse, bounds_error=False)
        # return 180 - np.abs(phaseResponseInterpolated(ugf))
        return 180 - np.abs(np.interp(ugf, frequenciesInHertz, phaseResponse, left=np.nan, right=np.nan))
    else:
        raise CalculationError("impossible to calculate the phase margin, either because this circuit never reaches unity gain (which means PM makes no sense) or your simulation data is insufficient. Try simulating with wider frequency range.")

def gainMargin(frequenciesInHertz, frequencyResponse):
    """Calculate the gain margin (not in dB)

    Attributes
    ----------

    frequenciesInHertz : 1-D ndarray
        Frequency points in Hz
    frequencyResponse : 1-D ndarray
        Frequency response points, given as an array of complex numbers
    """
    amplitudeResponse = np.abs(frequencyResponse)
    # amplitudeResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz, amplitudeResponse, bounds_error=False)
    return 1 - np.interp(positiveFeedbackFrequency(frequenciesInHertz, frequencyResponse), frequenciesInHertz, amplitudeResponse)

def gain(frequenciesInHertz, frequencyResponse):
    """Calculate the gain at `start` Hz, return as a complex number
    """
    try:
        # return scipy.interpolate.interp1d(frequenciesInHertz, frequencyResponse)(1)
        # return np.interp(1, frequenciesInHertz, frequencyResponse)

        # NOTE: Leme, 2012:
        # let av0 = gainvec[0]
        return frequencyResponse[0]
    except:
        raise CalculationError("impossible to calculate the DC gain because the data does not contain gain at {} Hz.".format(frequenciesInHertz[0]))


# NOTE: updated Slew Rate calculator.
def slewRate(timeInSecond, wave):
    r"""Calculate the slew rate by Moreto, 2024-05-23's definition.

    Notes
    -----

    SPICE code:

    let c1=0
    cursor c1 right time 12.5u
    let out1=v(out)[%c1]

    let c2=0
    cursor c2 right time 17.5u
    let out2=v(out)[%c2]

    let p10=(out2-out1)/10
    let out10=out1+p10
    let out90=out2-p10

    let c1=0
    let c2=0
    let srr_v_us=0

    * Move cursor c1 to 10% of v(out) level at third edge
    * (beginning of the second pulse cycle)
    cursor c1 right v(out) out10 3
    * Move cursor c2 to 90% of v(out) level at third edge
    * (beginning of the second pulse cycle)
    cursor c2 right v(out) out90 3

    let trise=(time[%c2]-time[%c1])*1E6

    if trise ne 0
        srr_v_us=(out90-out10)/trise
    end
    """
    # TODO: get it from the second pulse
    # (will need to change the pulse as well):
    # t_lwr = 12.5e-6
    # t_upr = 17.5e-6

    # pulse(0.0 2.0 5u 1p)
    # tran  0.1us 25us
    t_lwr = 2.5e-6
    t_upr = 22.5e-6

    sr = 0

    try:
        i_lwr = floatCursorRight(timeInSecond, t_lwr)
        i_upr = floatCursorRight(timeInSecond, t_upr)

        out1 = floatCursorGet(wave, i_lwr)
        out2 = floatCursorGet(wave, i_upr)

        p10 = (out2 - out1) / 10
        out10 = out1 + p10
        out90 = out2 - p10

        i10 = floatCursorRight(wave, out10)
        i90 = floatCursorRight(wave, out90)

        # NOTE: use interpolated wave[i90] - wave[i10] values,
        # to prevent misleading results when interpolation fails;
        # in case the amplitude never reaches the thresholds.
        out10 = floatCursorGet(wave, i10)
        out90 = floatCursorGet(wave, i90)

        t10 = floatCursorGet(timeInSecond, i10)
        t90 = floatCursorGet(timeInSecond, i90)

        trise = t90 - t10
        assert trise > 0
        sr = (out90 - out10) / trise
    except:
        print("slewRate undefined, vout range:", f"[{np.min(wave)}, {np.max(wave)}]")
        # NOTE: uncomment below for verbose output.
        # print("time:", timeInSecond, end=", ")
        # print("wave:", wave)
        # input("Press Enter to continue...")

    return sr


# NOTE: naive definition (possibly unstable)
def slewRateNaive(timeInSecond, wave):
    r"""Calculate the slew rate by naive definition

    Notes
    -----

    There exists huge ambiguity about what really is slew rate. According to Wikipedia, slew rate stands for the maximum absolute value of the output's derivative to time:

    .. math::

        SR = \max\left|{dV_o \over dt}\right|

    However, in some context, slew rate means the 2 thresholds (often 10% of delta and 90% of delta) divided by the time it takes the wave to rise from the low threshold to the high threshold. For example, consider a wave that travels from 1 V to 2 V. The slew rate is sometimes considered as (1.9 - 1.1) divided by the time it takes the wave to go up from 1.1 V to 1.9 V. If the duration is 1 s, then slew rate is 0.8/1 = 0.8 V/s.
    """
    return np.max(np.abs(np.diff(wave) / np.diff(timeInSecond)))


def slewRateLeme(timeInSecond, wave, lwr, upr):
    r"""Calculate the slew rate by Leme, 2012's definition.

    Notes
    -----
    SPICE code:

    let tupr = v(4) gt v(upr)
    let tlwr = v(4) lt v(lwr)

    if (tupr | tlwr)
        let sr = -sum(abs(v(4) * tupr) + abs(v(4) * tlwr)) * 1e6
    else
        let out10=0.3
        let out90=1.5

        let c1=0
        let c2=0
        cursor c1 right V(4) out10 1
        cursor c2 right V(4) out90 1

        let trise = time[%c2] - time[%c1]
        let sr = (out90 - out10) / trise
    end
    """
    t_upr = wave > upr
    t_lwr = wave < lwr

    if any(t_upr | t_lwr):
        sr = np.sum(np.abs(wave * t_upr) + np.abs(wave * t_lwr)) * 1e6
    else:
        out10 = 0.3
        out90 = 1.5

        # real-valued index to out10
        i10 = floatCursorRight(wave, out10)
        # real-valued index to out90
        i90 = floatCursorRight(wave, out90)

        t10 = floatCursorGet(timeInSecond, i10)
        t90 = floatCursorGet(timeInSecond, i90)

        trise = t90 - t10
        assert trise > 0

        # TODO: maybe return 0 if the thresholds are not met.
        sr = (out90 - out10)/trise

    return sr


def risingTime(timeInSecond, wave, threshold1=None, threshold2=None):
    """Measure the time it takes the wave to increase from `threshold1` to `threshold2` for the first time.

    Attributes
    ----------

    timeInSecond : time sequence
    wave : wave sequence
    threshold1 : low threshold
    threshold2 : high threshold

    Note
    ----

    It will not check whether threshold2 is greater than threshold1.
    """
    threshold1 = threshold1 or np.min(wave)
    threshold2 = threshold2 or np.max(wave)
    index1 = conditionFirstOccurrenceIndex(wave, wave > threshold1)
    index2 = conditionFirstOccurrenceIndex(wave, wave > threshold2)
    interpolater1 = scipy.interpolate.interp1d(timeInSecond[index1 - 1: index1 + 1], wave[index1 - 1: index1 + 1], bounds_error=False)
    interpolater2 = scipy.interpolate.interp1d(timeInSecond[index2 - 1: index2 + 1], wave[index2 - 1: index2 + 1], bounds_error=False)
    time1 = scipy.optimize.root(lambda x: interpolater1(x) - threshold1, timeInSecond[index1 - 1]).x[0]
    time2 = scipy.optimize.root(lambda x: interpolater2(x) - threshold2, timeInSecond[index2 - 1]).x[0]
    return time2 - time1

def fallingTime(timeInSecond, wave, threshold1=None, threshold2=None):
    """Measure the time it takes the wave to decrease from `threshold1` to `threshold2` for the first time.

    Attributes
    ----------

    timeInSecond : time sequence
    wave : wave sequence
    threshold1 : high threshold
    threshold2 : low threshold

    Note
    ----

    It will not check whether threshold1 is greater than threshold2.
    """
    threshold1 = threshold1 or np.max(wave)
    threshold2 = threshold2 or np.min(wave)
    index1 = conditionFirstOccurrenceIndex(wave, wave < threshold1)
    index2 = conditionFirstOccurrenceIndex(wave, wave < threshold2)
    interpolater1 = scipy.interpolate.interp1d(timeInSecond[index1 - 1: index1 + 1], wave[index1 - 1: index1 + 1], bounds_error=False)
    interpolater2 = scipy.interpolate.interp1d(timeInSecond[index2 - 1: index2 + 1], wave[index2 - 1: index2 + 1], bounds_error=False)
    time1 = scipy.optimize.root(lambda x: interpolater1(x) - threshold1, timeInSecond[index1 - 1]).x[0]
    time2 = scipy.optimize.root(lambda x: interpolater2(x) - threshold2, timeInSecond[index2 - 1]).x[0]
    return time2 - time1


def floatCursorRight(wave, target):
    # index where wave is closest to target.
    i = np.argmin(abs(wave - target))

    # local region
    i0 = max(i - 1, 0)
    i1 = min(i + 2, len(wave))
    xp = wave[i0:i1]
    # indices
    fp = range(i0, i1)

    diff_x = np.diff(xp)

    # xp must be monotonic.
    assert copysign(diff_x[0], diff_x[-1]) == diff_x[0], f"i: {i}, xp: {xp}"

    # xp must be increasing
    xp = np.copysign(xp, diff_x[0])
    # target inverts when xp was originally decreasing
    target = copysign(target, diff_x[0])

    # TODO: evaluate other forms of interpolation, and compare
    # them to what the cursor command does in SPICE.
    return np.interp(target, xp, fp)


# TODO: make sure this is better:
# left=0, right=0
# (return 0 when out of bounds, i.e.,
# x < xp[0] or x > xp[-1])
def floatCursorGet(wave, cursor, left=0, right=0):
    i0 = floor(cursor)
    i1 = ceil(cursor)

    # local region
    # [i0, i1]
    fp = wave[i0:i1 + 1]
    xp = range(max(i0, 0), min(i1 + 1, len(wave)))

    return np.interp(cursor, xp, fp, left=left, right=right)
