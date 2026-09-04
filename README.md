Dataset Setup

This project uses [DVC](https://dvc.org/) with DagsHub Storage for dataset versioning.

1. Clone the repository
```
git clone https://github.com/aircokezero/Cloud-Edge-Tradeoff.git
cd Cloud-Edge-Tradeoff
```

2. Create and activate a virtual environment
```
python -m venv .venv
```

Windows (Git CMD):
```
call .venv\Scripts\activate.bat
```

3. Install DVC with S3 support
```
python -m pip install --upgrade pip
pip install "dvc[s3]" dagshub
```

4. Configure the DagsHub DVC remote
```
dvc remote add -d storage s3://dvc
dvc remote modify storage endpointurl https://dagshub.com/aircokezero/Cloud-Edge-Tradeoff.s3
```

5. Configure DagsHub authentication

Generate a DagsHub access token and use it for both credentials:
```
dvc remote modify --local storage access_key_id YOUR_DAGSHUB_TOKEN
dvc remote modify --local storage secret_access_key YOUR_DAGSHUB_TOKEN
```

6. Download the datasets
```
dvc pull
```

The datasets will be restored to the `data/` directory.
> **Note:** Never commit your DagsHub access token. The `--local` flag stores the credentials locally in `.dvc/config.local`.

Updating Datasets
After modifying or adding files inside `data/`:
```
dvc add data
git add data.dvc
git commit -m "Update datasets"
dvc push
git push origin main
git push dagshub main
```
