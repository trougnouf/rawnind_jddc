mkdir -p src/flat
cd src/flat
curl -s "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM" | jq -r '.data.latestVersion.files[] | "wget -c -O \"\(.dataFile.filename)\" https://dataverse.uclouvain.be/api/access/datafile/\(.dataFile.id)"' | bash
