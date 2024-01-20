#!/bin/bash

# Download the data for Alanine Dipeptide.
# See https://github.com/lollcat/fab-torch for details.
curl "https://zenodo.org/api/records/6993124/files/val.pt/content" --output "data/aladip_val.pt"
