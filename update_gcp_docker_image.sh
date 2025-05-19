docker build -t pangaea .
docker tag pangaea:latest us-west1-docker.pkg.dev/eofm-benchmark/pangaea-bench/pangaea:latest
docker push us-west1-docker.pkg.dev/eofm-benchmark/pangaea-bench/pangaea:latest