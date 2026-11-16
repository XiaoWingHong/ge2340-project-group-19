# Label Studio config

This directory holds the labeling interface and a compose file for a local
Label Studio service. It does not start or change any already-running
deployment.

- `labeling_config.xml` — six-class taxonomy: promo, admin, course, personal, news, other
- `docker-compose.yml.example` — copy to `docker-compose.yml` (default host port 8090). The live compose file is gitignored.
- `sample_task.json` — example task shape for a manual import check

Create a project, paste `labeling_config.xml` into the labeling setup, and
import tasks produced by `converter/` (JSON array of
`{ "id", "data": { "from", "subject", "body" } }`).

```sh
# optional local stack; do not use this to mutate an existing deployment
cp docker-compose.yml.example docker-compose.yml
docker compose up -d
```

Keep tokens and live credentials out of git (see the root `.gitignore`).
