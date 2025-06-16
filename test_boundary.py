import sys
sys.path.insert(0, 'cora_python')
from cora_python.contSet.interval.interval import Interval

I = Interval([-2, -1], [3, 4])

# Test different points
test_points = [
    ([0, 0], "inside"),
    ([3, 4], "boundary (sup)"),
    ([-2, -1], "boundary (inf)"),
    ([4, 0], "outside"),
    ([-3, 0], "outside")
]

print(f"Interval: inf={I.inf}, sup={I.sup}")

for point, desc in test_points:
    result = I.contains(point)
    print(f"I.contains({point}) = {result} ({desc})")
    
    # Debug the boundary case specifically
    if desc == "boundary (sup)":
        print(f"  Point: {point}")
        print(f"  I.inf <= point: {I.inf <= point}")
        print(f"  point <= I.sup: {point <= I.sup}")
        print(f"  Manual check: {I.inf[0] <= point[0] <= I.sup[0] and I.inf[1] <= point[1] <= I.sup[1]}") 