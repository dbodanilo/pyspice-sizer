import scipy.optimize
import scipy.interpolate
import numpy as np

import functools

from math import copysign


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
    """
    amplitudeResponse = np.abs(frequencyResponse)
    try:
        # firstBelowUnityIndex = np.min(np.where(amplitudeResponse < 1))

        # amplitudeResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz[firstBelowUnityIndex - 1: firstBelowUnityIndex + 1], amplitudeResponse[firstBelowUnityIndex - 1: firstBelowUnityIndex + 1], bounds_error=False)
        # amplitudeResponseInterpolated = scipy.interpolate.interp1d(frequenciesInHertz, amplitudeResponse, bounds_error=False)

        # return scipy.optimize.root(lambda x: np.interp(x, \
        # frequenciesInHertz[firstBelowUnityIndex - 1: firstBelowUnityIndex + 1], \
        # amplitudeResponse[firstBelowUnityIndex - 1: firstBelowUnityIndex + 1], \
        # left=np.nan, right=np.nan) - 1, frequenciesInHertz[firstBelowUnityIndex - 1]).x[0]

        # closest amplitude to 1.
        # let cur = 0
        # cursor cur right gainvec 1
        i = np.argmin(abs(amplitudeResponse - 1))

        # local region:
        # [i - 1, i, i + 1]
        xp = amplitudeResponse[i - 1:i + 2]
        fp = frequenciesInHertz[i - 1:i + 2]

        diff_x = np.diff(xp)

        # xp must be monotonic.
        assert copysign(diff_x[0], diff_x[-1]) == diff_x[0]

        # xp must be increasing.
        xp = np.copysign(xp, diff_x[0])
        # unity gain turns to -1 when amplitude decreases.
        x = np.copysign(1, diff_x[0])

        # return 0 when out of bounds
        # (x < xp[0] or x > xp[-1]).
        # let fT = real(frequency[%cur])
        return np.interp(x, xp, fp, left=0, right=0)
    except:
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

        print(f"closest amplitude ({name}):", amplitudeResponse[i], end=", ")
        print("frequency:", frequenciesInHertz[i])

        return 0
        # raise CalculationError("impossible to calculate the unity gain frequency, because the data contains no amplitude point that is less than or equals 1. Try simulating with wider frequency range, or this circuit does not reach unity gain at all.")

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
    # except:
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

# TODO: update Slew Rate calculator.
def slewRate(timeInSecond, wave, lwr, upr):
    r"""Calculate the slew rate by naive definition

    Notes
    -----

    There exists huge ambiguity about what really is slew rate. According to Wikipedia, slew rate stands for the maximum absolute value of the output's derivative to time:

    .. math::

        SR = \max\left|{dV_o \over dt}\right|
    
    However, in some context, slew rate means the 2 thresholds (often 10% of delta and 90% of delta) divided by the time it takes the wave to rise from the low threshold to the high threshold. For example, consider a wave that travels from 1 V to 2 V. The slew rate is sometimes considered as (1.9 - 1.1) divided by the time it takes the wave to go up from 1.1 V to 1.9 V. If the duration is 1 s, then slew rate is 0.8/1 = 0.8 V/s.
    """
    # naive definition (possibly unstable)
    # return np.max(np.abs(np.diff(wave) / np.diff(timeInSecond)))

    # let tupr = v(4) gt v(upr)
    t_upr = wave > upr
    # let tlwr = v(4) lt v(lwr)
    t_lwr = wave < lwr
    # if (tupr | tlwr)
    if any(t_upr | t_lwr):
        # let st = -sum(abs(v(4) * tupr) + abs(v(4) * tlwr)) * 1e6
        sr = -np.sum(np.abs(wave * t_upr) + np.abs(wave * t_lwr)) * 1e6
    else:
        out10 = 0.3
        out90 = 1.5

        # closest wave value to out10
        i10 = np.argmin(abs(wave - out10))
        # closest wave value to out90
        i90 = np.argmin(abs(wave - out10))

        # local region
        # [i10 - 1, i10, i10 + 1]
        fp10 = wave[i10 - 1:i10 + 2]
        fp90 = wave[i90 - 1:i90 + 2]

        xp10 = range(np.max(i10 - 1, 0), np.min(i10 + 2, len(wave)))
        xp90 = range(np.max(i90 - 1, 0), np.min(i90 + 2, len(wave)))

        t10 = timeInSecond[i10 - 1:i10 + 2]
        t90 = timeInSecond[i90 - 1:i90 + 2]

        # real-valued indices
        # cursor c1 right V(4) out10
        i10 = np.interp(out10, xp10, fp10)
        # cursor c2 right V(4) out90
        i90 = np.interp(out90, xp90, fp90)

        # time[%c1]
        t10 = np.interp(i10, xp10, t10)
        # time[%c2]
        t90 = np.interp(i90, xp90, t90)

        # let sr = (out90 - out10) / (time[%c2] - time[%c1])
        # TODO: use interpolated wave[i90] - wave[i10] values,
        # to prevent misleading results when interpolation fails;
        # even return 0 if the thresholds are not met.
        sr = (out90 - out10)/(t90 - t10)

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