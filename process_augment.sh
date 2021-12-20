#!/bin/bash
rm -r aug_example/
python apply_augm.py -i sound/fan -o aug_example/fan
python apply_augm.py -i sound/pump -o aug_example/pump
python apply_augm.py -i sound/slider -o aug_example/slider
python apply_augm.py -i sound/ToyCar -o aug_example/ToyCar
python apply_augm.py -i sound/valve -o aug_example/valve
python apply_augm.py -i sound/ToyConveyor -o aug_example/ToyConveyor