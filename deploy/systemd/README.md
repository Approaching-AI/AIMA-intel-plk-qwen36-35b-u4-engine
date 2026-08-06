# systemd deployment

The unit assumes the installed source tree and virtual environment live under
`/opt/intel-qwen36`, the model and generated runtime bundle live under
`/opt/iq36`, and a dedicated unprivileged `iq36` account owns the compile
cache. Before installation, build the immutable runtime bundle:

```bash
python3 tools/intel-qwen36-package-runtime-assets.py \
  --output /opt/iq36/runtime
```

The command refuses to overwrite an existing directory. Inspect and retain
`/opt/iq36/runtime/manifest.json` with the deployment record.

Build the service wheel and exact offline Python wheelhouse on the bound build
target, then install it into the deployment venv:

```bash
SOURCE_DATE_EPOCH=315532800 \
  /home/intel/ov/openvino_env/bin/python -m pip wheel . --no-deps \
  --wheel-dir output/http-service-dist
python3 tools/intel-qwen36-package-python-runtime.py \
  --python /home/intel/ov/openvino_env/bin/python \
  --service-wheel output/http-service-dist/intel_qwen36_server-0.1.0-py3-none-any.whl \
  --output output/http-python-wheelhouse
python3.12 -m venv /opt/intel-qwen36/.venv
/opt/intel-qwen36/.venv/bin/python -m pip install \
  --no-index --find-links=output/http-python-wheelhouse pip==26.2.1
/opt/intel-qwen36/.venv/bin/python -m pip install \
  --no-index --find-links=output/http-python-wheelhouse \
  --require-hashes \
  -r output/http-python-wheelhouse/bound-runtime-requirements.txt
/opt/intel-qwen36/.venv/bin/python -m pip check
```

Retain the wheelhouse `manifest.json` with the deployment record. Startup
rejects a different OpenVINO or GenAI build even when its package release
number appears compatible.

The unit grants the service account the host's `render` and `video`
supplementary groups so it can open `/dev/dri/renderD*`. If the distribution
uses different GPU-device groups, adjust `SupplementaryGroups` before enabling
the unit; do not run the service as root to work around device permissions.

Example installation (run with the appropriate administrative privileges):

```bash
useradd --system --home-dir /nonexistent --shell /usr/sbin/nologin iq36
install -d -m 0750 -o root -g iq36 /etc/iq36
install -m 0640 -o root -g iq36 deploy/iq36.env.example /etc/iq36/iq36.env
install -m 0644 deploy/systemd/iq36.service /etc/systemd/system/iq36.service
```

Create the bearer key without placing it in shell history:

```bash
umask 077
openssl rand -hex 32 | tee /etc/iq36/api-key >/dev/null
chown root:root /etc/iq36/api-key
chmod 0600 /etc/iq36/api-key
```

Edit `/etc/iq36/iq36.env`, verify the external model and the generated asset
manifest against the provenance record, then enable the unit:

```bash
systemctl daemon-reload
systemctl enable --now iq36.service
systemctl status iq36.service
curl http://127.0.0.1:8000/readyz
```

The readiness endpoint is intentionally unauthenticated for a local process
supervisor. API routes and `/metrics` require the bearer key. If the listener
is exposed beyond loopback, put it behind a TLS reverse proxy and retain both
network-level rate limiting and the service's bearer authentication. The
shipped unit also applies a systemd IP allow-list for loopback; remove or
replace `IPAddressDeny`/`IPAddressAllow` deliberately if the reverse proxy is
on another host.

`TimeoutStopSec` is longer than `IQ36_SHUTDOWN_TIMEOUT`: the controller first
stops accepting connections, drains admitted requests, then cancels remaining
work and shuts down the worker process group.

`TimeoutStartSec` includes the default full SHA-256 pass over the locked model,
tokenizer initialization, graph compilation, and shape warmup. Keep
`IQ36_MODEL_VERIFICATION=full` for production; a metadata-only or disabled
identity check is diagnostic and is exposed as such by `/readyz`.
