# FeatureExtractorStacking

> [**Feature Extractor Stacking for Cross-domain Few-shot Meta-learning**](https://arxiv.org/abs/2205.05831),            
> Hongyu Wang, Eibe Frank, Bernhard Pfahringer, Michael Mayo, Geoffrey Holmes

The quickest way to verify the main results is to use cached logits, which can be downloaded from Zenodo:

[FES-TSA source domains](https://zenodo.org/record/7309028)

[FES-TSA source domains](https://zenodo.org/record/7308984)

[FES-URL source domains](https://zenodo.org/record/7309165)

[FES-URL target domains](https://zenodo.org/record/7309091)

[FES-FLUTE all domains](https://zenodo.org/record/7312066)

In the root directory "FeatureExtractorStacking/", please create a directory named "dataset/", and create three child directories for the fine-tuning algorithms "tsa/", "url/", and "flute/". Uncompress the downloaded data into these directories. Every cached episode should have a path such as "FeatureExtractorStacking/datasets/tsa/traffic_sign/0.pt".

Please install the environment with anaconda

    ```
    conda env create -f environment.yml
    ```

To test ConFES and ReFES on the cached episodes, please run

    ```
    python test.py
    ```

Fine-tuning algorithms can be selected by modifying the `DATASETS_DIR` string in "test.py", which uses TSA logits by default.

In order to make our cached data available, we had to store the logits in half-precision floats. Therefore, the results obtained here may differ slightly from the ones in the paper. We can provide our original full-precision cache if requested.

To only run a subset of the 600 episodes for a quicker demonstration, please edit the dataset length in the "test.py" file, from

    ```
    def __len__(self):
    
        return len(os.listdir(self.dataset_dir))
    ```

to (for example, if we only want 5 episodes)

    ```
    def __len__(self):
    
        return 5
    ```

To plot heatmaps for an episode, please run

    ```
    python plot.py
    ```

The plots will be saved in "heatmaps/". The episode can be selected by editing "plot.py":

    ```
    CACHEFILENAME = 'datasets/tsa/traffic_sign/0.pt'
    ```

We are actively working on code that performs fine-tuning and FES on few-shot episodes with raw images, and saves snapshots and meta-classifiers in files for later practical use.
