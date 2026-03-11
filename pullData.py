from pathlib import Path

import synapseclient
from synapseutils import syncFromSynapse



syn = synapseclient.Synapse()
syn = synapseclient.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3MzE1ODk3NiwiaWF0IjoxNzczMTU4OTc2LCJqdGkiOiIzMzUxOSIsInN1YiI6IjM1NzgzOTIifQ.HY1sy6UdS9hQTGL497aRDd4n5IOsFhpt03CJp_AdZ7I7nHZuBiPm6vgDs6xqmek7GaFuElW9M-ySBq1_i9ymr9ws_ltFHCkVWvmF93aLI9E-2tOD_0ck6aK11CKw0sKUVe7C7CJMZG7VRt8eFUkWk_Jn8Qt2oMIKIYWeCpZaKxLFLyNQzf7xIPVOOXozdVDgtm-rc6x8cZZD-GPcRT9zkD63URTlIkQcG64fku0uKD4j5cJnZMdaNrHsI4qR46GYAcDeA_uhC6fX2Zf69G7QPbQ8-6ke62m-WewYe5uZkpH4LF4_fUz5uq5eQH1-Q800w4u_nmMbnqrwbodzk9EpNg")


download_dir = Path("data")
download_dir.mkdir(parents=True, exist_ok=True)

files = syncFromSynapse(syn, 'syn11853680', path=str(download_dir))
print(f"Downloaded/checked {len(files)} items into: {download_dir.resolve()}")
for item in files[:10]:
    print(item)
if not files:
    print("No file entities were downloaded. Check that this Synapse ID contains files you can access.")
