SA-PatchCore Implementation
This repository contains an implementation of SA-PatchCore for anomaly detection.
## Dataset Preparation
~~~
0424/
└── patchcore_13_V2/
    ├── train/
    │   └── 0_good/
    │       ├── img1.png
    │       ├── img2.png
    │       └── ...
    └── test/
        ├── 0_good/
        │   ├── img1.png
        │   └── ...
        └── 1_scratch/
            ├── img1.png
            └── ...
~~~
## Set Hyperparameters
Specify defect classes (Line 263)
~~~
name_defect_types = ['0_good', '1_scratch']
~~~
2. Configure arguments (Line 311)
~~~
def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'C:\Users\user\Desktop\0424\patchcore_13_V2') 
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--input_size', default=512)
    parser.add_argument('--coreset_sampling_ratio', default=0.05)
    parser.add_argument('--output_path', default=r'./outputs') 
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args
~~~
## train(run.py)
After setting the hyperparameters, you can train the model by running:
## test(patchcore_test_alldata.py)
Select Folder (line 163)
~~~
if __name__ == '__main__':
    dataset_path = r'C:\Users\user\Desktop\20241216_ITRI'
    output_path = r'C:\Users\user\Desktop\0424\patchcore_result\13_1024_0.1'
    checkpoint_path = os.path.join(output_path, 'lightning_logs', 'version_0', 'checkpoints', 'epoch=0-step=13.ckpt')
~~~
