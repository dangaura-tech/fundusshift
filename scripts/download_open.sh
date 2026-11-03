#!/usr/bin/env bash
# Download every redistributable source into <dl_dir>/<source_id>/ (archives only; extraction is separate).
# Usage: scripts/download_open.sh <dl_dir>
set -uo pipefail
DL=${1:?download dir}
get() {  # id url filename
  mkdir -p "$DL/$1"
  curl -sSL --retry 5 --retry-delay 10 -C - -o "$DL/$1/$3" "$2" && echo "done $1/$3" || echo "FAILED $1/$3"
}
if [ "${V11:-0}" != "1" ]; then
get idrid        "https://zenodo.org/api/records/17219542/files/B.%20Disease%20Grading.zip/content" grading.zip &
get rfmid2       "https://zenodo.org/api/records/7505822/files/RFMiD2_0.zip/content" RFMiD2_0.zip &
get paraguay_dr  "https://zenodo.org/api/records/4891308/files/Dataset%20from%20fundus%20images%20for%20the%20study%20of%20diabetic%20retinopathy_V03.zip/content" paraguay.zip &
get fives        "https://ndownloader.figshare.com/files/34969398" fives.rar &
get papila       "https://ndownloader.figshare.com/files/35013982" PAPILA.zip &
get sustech_sysu "https://ndownloader.figshare.com/files/25320596" sustech.zip &
get jichi        "https://ndownloader.figshare.com/files/8169134" dmr.tar &
get grape        "https://ndownloader.figshare.com/files/41358156" CFPs.rar &
get hrf          "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip" all.zip &
get kim_glaucoma "https://dataverse.harvard.edu/api/access/datafile/3314942" processed_data.zip &
get hygd         "https://physionet.org/content/hillel-yaffe-glaucoma-dataset/get-zip/1.1.0/" hygd.zip &
get deepdrid     "https://github.com/deepdrdoc/DeepDRiD/archive/refs/heads/master.zip" deepdrid.zip &
get bangladesh_eye "https://data.mendeley.com/public-files/datasets/s9bfhswzjb/files/13267974-54bf-42b0-87cb-f691b48ba384/file_downloaded" original.zip &
wait
get chaksu       "https://ndownloader.figshare.com/files/38944805" readme.pdf
get chaksu       "https://ndownloader.figshare.com/files/37875687" Test.zip &
get chaksu       "https://ndownloader.figshare.com/files/37875672" Train.zip &
wait
echo ALL_DONE
fi

# ---- v1.1 additions ----
if [ "${V11:-0}" = "1" ]; then
get acrima        "https://ndownloader.figshare.com/files/14137700" Database.zip &
get palm          "https://ndownloader.figshare.com/files/37786152" PALM.zip &
get rajbari_eye   "https://data.mendeley.com/public-files/datasets/z227d626vm/files/9c96f4d4-6041-47e5-9d13-a230dbe9847f/file_downloaded" retinal.zip &
get afio_composite "https://data.mendeley.com/public-files/datasets/trghs22fpg/files/a5961cce-bc62-4a21-afe8-575261aa0c08/file_downloaded" dataset.zip &
get afio_hr       "https://data.mendeley.com/public-files/datasets/3csr652p9y/files/5c07e45a-5f3f-407b-8bdb-16332a84fa23/file_downloaded" av.zip &
wait
echo V11_DONE
fi
# DDR (CC BY-NC-SA 4.0): Google Drive folder 1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC, 10 chunks; download with gdown,
# then: cat DDR-dataset.zip.0* > DDR-dataset.zip
