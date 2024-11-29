# ASDnB: Merging Face with Body Cues For Robust Active Speaker Detection

1. Download the AVA-ActiveSpeaker [\[github\]](https://github.com/cvdfoundation/ava-dataset) [\[website\]](https://research.google.com/ava/download.html#ava_active_speaker_download) and [WASD](https://tiago-roxo.github.io/WASD/) datasets;
2. Create the environment with `conda` using the script `create_env.sh`;
3. To train the model, execute the command:
```bash
python3 main.py --dataPath $DATASET
```
where `$DATASET` is the folder containing the dataset (AVA-ActiveSpeaker or WASD). Alternatively, you can run the script `train.sh` which contains an example of training the model in WASD. After each epoch, the model weights will be saved and output its predictions to a CSV (`val_res.csv`), in the `exps/wasd` folder (by default);

4. To evaluate the model, execute the command:
```bash
python3 main.py --dataPath $DATASET --evaluation
```
where `$DATASET` is the folder containing the dataset (AVA-ActiveSpeaker or WASD). Alternatively, you can run the script `infer.sh` which contains an example of evaluating the model in WASD. The model predictions are saved in a CSV (`val_res.csv`), in the `exps/wasd` folder (by default).

The ASDnB model **trained on WASD** can be downloaded [here]().

The ASDnB model **trained on AVA-ActiveSpeaker** can be downloaded [here]().

## Columbia

The results for [Columbia dataset](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_18) were obtained using the evaluation tool available at [Light-ASD](https://github.com/Junhua-Liao/Light-ASD). You can obtain the dataset already preprocessed with face and body bounding boxes [here](https://drive.google.com/file/d/1nZoMoTq_bmMl1PiPttmbuL_oC19jhVrR/view?usp=sharing).

To evaluate the model, execute the command:
```bash
python3 Columbia_infer.py --videoFolder $DATASET
```
where `$DATASET` is the folder containing the dataset (Columbia). Alternatively, you can run the script `infer_Columbia.sh` which contains an example of evaluating the model in Columbia.

The ASDnB model **for Columbia** can be downloaded [here]().

## Cite

```bibtex

```