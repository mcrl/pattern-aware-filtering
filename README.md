# Pattern-Aware Line-Filtering

A collection of tools for extracting Korean and English text for LLM training from the CommonCrawl WET dataset.

Codes for **Beyond Line-Level Filtering for the Pretraining Corpora of LLMs**

## System Requirements

Our data pipeline were run with 64 CPU vCores and 512GB RAM space.
This tool requires a system with more than 400GB CPU RAM space to function stablely.
For systems with lower RAM spaces, you may reduce the degree of parallelism, yet you need 50GB of RAM space at least, even for the single process settings.

## Installation

### Command line tools

```plain
wget
gunzip
```

### Installing package

```bash
pip install -e .

# downloading fasttext language identification model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -O pattern_aware_filtering/utils/lid.176.ftz
```

## Overview

Our scripts contain following steps

1. Preparing baseline data, including downloads of CC WET files.
2. Building hashmap for obtaining the `count`, which is the number of the document that contains a text line in a document set.
3. Obtaining intermediate data: `count info` file that contains (i) the raw data and (ii) the count information for each text line of the raw data.
> This intermediate data allows ease of extracting text with Pattern-Aware Line-Level Deduplication with varying categorization thresholds.
4. Filtering the data with the intermediate data.

## Preparation

### Setup Constants

Fix data save path defined in  `pattern_aware_filtering/utils/constants.py` file for your purpose.

### Obtaining CommonCrawl WET Files
```bash
python scripts/prepare-ccnet/download_wet_files.py
```

### Obtaining list of CommonCrawl WET Files (Necessary)

```bash
python scripts/prepare-ccnet/download_wet_paths_file.py
```

### Sampling Splits

For handling English data, we did not extract all data from English shard, because of storage constraints.
We first sample which shards to be used and then selectively extract English data only from the sampled shards.
This step ensures that every experiment uses the same raw data for the training/validation split. Run the sampling script to generate the shard lists for each language:

```bash
python scripts/prepare-ccnet/sample_splits.py en
python scripts/prepare-ccnet/sample_splits.py ko
```

## Extracting Baseline Data(Language Identification)

### English

For handling English data, we did not extract all data from English shard, because of storage constraints.
We only process 10% of the total data. See the script for more detail.

```bash
bash scripts/extract_baseline_files/extract_baseline_en.sh
```

### Korean

We extract all Korean documents from all shards. See the script for more detail.

```bash
bash scripts/extract_baseline_files/extract_baseline_ko.sh
```


## Hashing

### English

To ensure each process run not longer than 12 hours(Our cluster's maximum length of job allocation), we employ two-step strategy for hashing. Our goal of this step is acquiring a hashmap spanning across 1000 shards. See the script for the detail, and parallelization options.
```bash
bash scripts/hashing/hashing_en.sh
```

### Korean

The number of Korean documents spans from 15 million to 25 million documents per snapshot. We first make a hashmap for each shard file. Then, we merge all the hash files.

```bash
bash scripts/hashing/hashing_ko.sh
```

## Obtaining Intermediate File: `Count Info`

As hashmap searching is the most memory-consuming part, we obtain the count information first for further analysis.

```bash
bash scripts/count_info/extract_en_count_info.sh # English
bash scripts/count_info/extract_ko_count_info.sh # Korean
```
## Extraction

### English

Following scripts apply PLD / PTF at the intermediate file. See the script for details.

```bash
bash scripts/extract-en/run_pld_ablation.sh # for setting r/y/g threshold for PLD
bash scripts/extract-en/run_ptf_ablation.sh # for acquiring 'k' for PTF. We obtain our final version of data with this script.
```

### Korean

Following scripts apply PLD / PTF at the intermediate file. See the script for details.

```bash
bash scripts/extract-ko/run_pld_ablation.sh # for setting r/y/g threshold for PLD
bash scripts/extract-ko/run_ptf_ablation.sh # for acquiring 'k' for PTF. We obtain our final version of data with this script.
```