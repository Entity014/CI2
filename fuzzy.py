import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# input
distance_front = ctrl.Antecedent(np.arange(0, 100, 1), "distance_front")
distance_left = ctrl.Antecedent(np.arange(0, 100, 1), "distance_left")
distance_right = ctrl.Antecedent(np.arange(0, 100, 1), "distance_right")

# output
position_x = ctrl.Consequent(np.arange(-10, 10, 0.1), "position_x")
position_y = ctrl.Consequent(np.arange(-10, 10, 0.1), "position_y")

# function
distance_front["near"] = fuzz.trimf(distance_front.universe, [0, 0, 50])
distance_front["medium"] = fuzz.trimf(distance_front.universe, [0, 50, 100])
distance_front["far"] = fuzz.trimf(distance_front.universe, [50, 100, 100])

distance_left["near"] = fuzz.trimf(distance_left.universe, [0, 0, 50])
distance_left["medium"] = fuzz.trimf(distance_left.universe, [0, 50, 100])
distance_left["far"] = fuzz.trimf(distance_left.universe, [50, 100, 100])

distance_right["near"] = fuzz.trimf(distance_right.universe, [0, 0, 50])
distance_right["medium"] = fuzz.trimf(distance_right.universe, [0, 50, 100])
distance_right["far"] = fuzz.trimf(distance_right.universe, [50, 100, 100])

position_x["left"] = fuzz.trimf(position_x.universe, [-10, -5, 0])
position_x["center"] = fuzz.trimf(position_x.universe, [-5, 0, 5])
position_x["right"] = fuzz.trimf(position_x.universe, [0, 5, 10])

position_y["back"] = fuzz.trimf(position_y.universe, [-10, -5, 0])
position_y["center"] = fuzz.trimf(position_y.universe, [-5, 0, 5])
position_y["front"] = fuzz.trimf(position_y.universe, [0, 5, 10])

# Rule
# F : N L : N R : N -> X : C Y : F
# F : N L : N R : M -> X : L Y : F
# F : N L : N R : F -> X : L Y : F
# F : N L : M R : N -> X : R Y : F
# F : N L : M R : M -> X : C Y : F
# F : N L : M R : F -> X : L Y : F
# F : N L : F R : N -> X : R Y : F
# F : N L : F R : M -> X : R Y : F
# F : N L : F R : F -> X : C Y : F

# F : M L : N R : N -> X : C Y : C
# F : M L : N R : M -> X : L Y : C
# F : M L : N R : F -> X : L Y : C
# F : M L : M R : N -> X : R Y : C
# F : M L : M R : M -> X : C Y : C
# F : M L : M R : F -> X : L Y : C
# F : M L : F R : N -> X : R Y : C
# F : M L : F R : M -> X : R Y : C
# F : M L : F R : F -> X : C Y : C

# F : F L : N R : N -> X : C Y : B
# F : F L : N R : M -> X : L Y : B
# F : F L : N R : F -> X : L Y : B
# F : F L : M R : N -> X : R Y : B
# F : F L : M R : M -> X : C Y : B
# F : F L : M R : F -> X : L Y : B
# F : F L : F R : N -> X : R Y : B
# F : F L : F R : M -> X : R Y : B
# F : F L : F R : F -> X : C Y : B
rule1 = ctrl.Rule(distance_front["near"], position_y["front"])
rule2 = ctrl.Rule(distance_front["medium"], position_y["center"])
rule3 = ctrl.Rule(distance_front["far"], position_y["back"])
rule4 = ctrl.Rule(distance_left["near"] & distance_right["near"], position_x["center"])
rule5 = ctrl.Rule(distance_left["near"] & distance_right["medium"], position_x["left"])
rule6 = ctrl.Rule(distance_left["near"] & distance_right["far"], position_x["left"])
rule7 = ctrl.Rule(distance_left["medium"] & distance_right["near"], position_x["right"])
rule8 = ctrl.Rule(
    distance_left["medium"] & distance_right["medium"], position_x["center"]
)
rule9 = ctrl.Rule(distance_left["medium"] & distance_right["far"], position_x["left"])
rule10 = ctrl.Rule(distance_left["far"] & distance_right["near"], position_x["right"])
rule11 = ctrl.Rule(distance_left["far"] & distance_right["medium"], position_x["right"])
rule12 = ctrl.Rule(distance_left["far"] & distance_right["far"], position_x["center"])

localization_ctrl = ctrl.ControlSystem(
    [
        rule1,
        rule2,
        rule3,
        rule4,
        rule5,
        rule6,
        rule7,
        rule8,
        rule9,
        rule10,
        rule11,
        rule12,
    ]
)
localization = ctrl.ControlSystemSimulation(localization_ctrl)

localization.input["distance_front"] = 70
localization.input["distance_left"] = 90
localization.input["distance_right"] = 25

localization.compute()

print(f"Position X: {localization.output['position_x']}")
print(f"Position Y: {localization.output['position_y']}")

distance_front.view(sim=localization)
distance_left.view(sim=localization)
distance_right.view(sim=localization)
position_x.view(sim=localization)
position_y.view(sim=localization)
plt.show()
