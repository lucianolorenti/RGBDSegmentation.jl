# Benchmark
## Database installation
```bash
export DB_USERNAME=rgbd_segmentation
export DB_NAME=rgbd_segmentation_results
export DB_PASSWORD=rgbd_seg
sudo -u postgres bash -c "psql -c \"create database $DB_NAME;\""
sudo -u postgres bash -c "psql -c \"create user $DB_USERNAME with encrypted password '$DB_PASSWORD'\"";
sudo -u postgres bash -c "psql -c \"grant all privileges on database $DB_NAME to $DB_USERNAME\"";
```
