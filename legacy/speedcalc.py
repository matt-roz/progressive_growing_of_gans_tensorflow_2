from collections import OrderedDict
from datetime import timedelta

stages = [i for i in range(2, 11)]
sizes = [2**i for i in stages]
images_per_epoch = 30000
seconds = OrderedDict(
    [
        (2, 0*60 + 5),
        (3, 0*60 + 16),
        (4, 0*60 + 46),
        (5, 2*60 + 42),
        (6, 6*60 + 3),
        (7, 11*60 + 20),
        (8, 21*60 + 45),
        (9, 42*60 + 00),
        (10, 62*60 + 15),
    ]
)
"""
seconds = OrderedDict(
    [
        (2, 1*60 + 50),
        (3, 3*60 + 10),
        (4, 4*60 + 19),
        (5, 7*60 + 2),
        (6, 11*60 + 00),
        (7, 17*60 + 14),
        (8, 30*60 + 2),
        (9, 59*60 + 40),
        (10, 121*60 + 45),
    ]
)
"""

images_per_stage = 800000 * 2
seconds2 = OrderedDict(
    [
        (2, 3.25),
        (3, 4.6),
        (4, 7.15),
        (5, 14.2),
        (6, 25.5),
        (7, 39.0),
        (8, 64.0),
        (9, 121.2),
        (10, 226),
    ]
)


# adapt seconds
for index, stage in enumerate(stages):
    if index + 2 in seconds.keys():
        continue
    else:
        _factor = seconds[index + 1] / seconds[index]
        seconds[index + 2] = seconds[index + 1] * _factor

_duration = 0
for stage, duration in seconds.items():
    _duration += duration * 54
    print(f"stage={stage} \t "
          f"factor={seconds[stage] / seconds[max(stage-1, 2)]:.3f} \t"
          f"resolution={str(2**stage).zfill(4)} \t "
          f"images_per_sec={images_per_epoch/duration:.3f} \t"
          f"sec_per_kimg={duration/(images_per_epoch/1000):.3f} \t"
          f"epoch_duration={str(timedelta(seconds=duration))} \t"
          f"stage_duration={str(timedelta(seconds=duration*54))} \t"
          f"full_duration={str(timedelta(seconds=_duration))}")
    if stage == 10:
        _duration += duration * 54
        print(f"converge \t duration={str(timedelta(seconds=duration*54))} \t "
              f"full_duration={str(timedelta(seconds=_duration))}")

_duration = 0
for stage, sec_per_kimages in seconds2.items():
    images_per_sec = (1/sec_per_kimages) * 1000
    duration = images_per_stage/images_per_sec
    if stage == 2:
        duration /= 2
    _duration += duration
    print(f"stage={stage} \t "
          f"factor={seconds2[stage] / seconds2[max(stage-1, 2)]:.3f} \t"
          f"resolution={str(2**stage).zfill(4)} \t "
          f"images_per_sec={images_per_sec:.3f} \t"
          f"sec_per_kimg={sec_per_kimages:.3f} \t"
          f"stage_duration={str(timedelta(seconds=duration))} \t"
          f"full_duration={str(timedelta(seconds=_duration))}")
    if stage == 10:
        _duration += duration
        print(f"converge \t duration={str(timedelta(seconds=duration))} \t "
              f"full_duration={str(timedelta(seconds=_duration))}")
