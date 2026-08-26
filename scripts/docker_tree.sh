#!/usr/bin/env sh
# Print the running AMLGuard stack as the tree the project is organized into. Docker Compose has a
# flat service model (no true nested services), so the hierarchy is carried on labels:
#   com.docker.compose.project=protegrity   → the root
#   com.docker.compose.group=app|vendor-de|observability   → the branches
#   com.docker.compose.subgroup=langfuse (etc.)            → a sub-tree within a branch
# This script reads those labels off the running containers and renders the indented tree.
set -eu

PROJECT="${1:-protegrity}"
# Pipe-delimited (a status can contain spaces/parens/tabs). Empty subgroup stays an empty field,
# which awk -F'|' handles correctly (unlike tab-splitting via `read`, where empty fields collapse).
FMT='{{.Names}}|{{.Label "com.docker.compose.group"}}|{{.Label "com.docker.compose.subgroup"}}|{{.Status}}'

rows="$(docker ps --filter "label=com.docker.compose.project=${PROJECT}" --format "$FMT" 2>/dev/null || true)"
if [ -z "$rows" ]; then
    echo "${PROJECT}: (no running containers)"
    exit 0
fi

echo "${PROJECT}"
# Fixed branch order (app first) so the tree reads top-down rather than alphabetically.
for group in app vendor-de observability; do
    grows="$(printf '%s\n' "$rows" | awk -F'|' -v g="$group" '$2==g')"
    [ -z "$grows" ] && continue
    echo "├─ ${group}"

    # Direct children of the branch (no subgroup).
    printf '%s\n' "$grows" | awk -F'|' '$3==""{printf "│  ├─ %s  (%s)\n", $1, $4}' | sort

    # Sub-trees: each distinct non-empty subgroup, with its own children indented one level deeper.
    subs="$(printf '%s\n' "$grows" | awk -F'|' '$3!=""{print $3}' | sort -u)"
    for sub in $subs; do
        echo "│  ├─ ${sub}"
        printf '%s\n' "$grows" | awk -F'|' -v s="$sub" '$3==s{printf "│  │  ├─ %s  (%s)\n", $1, $4}' | sort
    done
done

# Opt-in services are profile-gated and only run when asked for, so they don't appear above by
# default. Note the local-model Ollama one so a reviewer knows it exists and how to start it, instead
# of assuming it is missing. (Default live-chat model is host Ollama; this is the in-stack option.)
if ! printf '%s\n' "$rows" | awk -F'|' '$1=="ollama"{found=1} END{exit !found}'; then
    echo "└─ (opt-in) app/ollama  — in-stack open-source model server, off by default."
    echo "     start it: docker compose --env-file .env -f docker/app/ui/compose.full.yml \\"
    echo "                 --profile local-model up -d ollama amlguard_app   (OLLAMA_URL=http://ollama:11434)"
fi
