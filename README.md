# HMM
Repository for EECS 738 Project Hidden M... Mod..

## Installing Dependencies

```
$HMM> pip3 install -r reqirements.txt

```

## Usage

### Training
```
python3 hmm_shake.py -t
```
This will read in the complete works of shakespeare and output to file the transition, observation, and initialization matrices.

### Generating Text
```
python3 hmm_shake.py -g <length>
```
This will read in the saved training matrices and use them to output the console a string of the given length.
