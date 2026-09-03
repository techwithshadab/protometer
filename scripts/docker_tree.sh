#!/usr/bin/env sh
# Print the running Protometer stack as the tree the project is organized into. Docker Compose has a
# flat service model (no true nested services), so the hierarchy is carried on labels:
#   com.docker.compose.project=protometer   → the root (the default `docker-up` app stack)
#   com.docker.compose.group=app|vendor-de|observability   → the branches
#   com.docker.compose.subgroup=langfuse (etc.)            → a sub-tree within a branch
# This script reads those labels off the running containers and renders the indented tree.
#
# The DEFAULT decoupled flow (`make shared-up && make docker-up`) runs the app as its own
# `protometer` project beside sibling projects `observability-shared`, `protegrity-shared`, and
# `botox-demo`. Pass a project name to inspect one of those, or `protegrity` for the legacy
# monolithic `docker-full` stack (where the group/subgroup branches below are populated).
set -eu

PROJECT="${1:-protometer}"
# Pipe-delimited (a status can contain spaces/parens/tabs). Empty subgroup stays an empty field,
# which awk -F'|' handles correctly (unlike tab-splitting via `read`, where empty fields collapse).
FMT='{{.Names}}|{{.Label "com.docker.compose.group"}}|{{.Label "com.docker.compose.subgroup"}}|{{.Status}}'

rows="$(docker ps --filter "label=com.docker.compose.project=${PROJECT}" --format "$FMT" 2>/dev/null || true)"
if [ -z "$rows" ]; then
    echo "${PROJECT}: (no running containers)"
    # In the decoupled flow the app project has no group labels, so it renders flat below. If the
    # user asked for the default and nothing matched, point at the sibling projects that may be up.
    if [ "$PROJECT" = "protometer" ]; then
        others="$(docker ps --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null \
                    | sort -u | grep -vx protometer | grep -vx '' || true)"
        [ -n "$others" ] && printf 'other running compose projects: %s\n' "$(printf '%s ' $others)"
    fi
    exit 0
fi

echo "${PROJECT}"

# Decoupled flow: the standalone app project carries no group labels, so render its containers as a
# flat list under the root and skip the branch machinery (which only the monolithic stack populates).
if ! printf '%s\n' "$rows" | awk -F'|' '$2!=""{found=1} END{exit !found}'; then
    printf '%s\n' "$rows" | awk -F'|' '{printf "├─ %s  (%s)\n", $1, $4}' | sort
    exit 0
fi

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
    echo "                 --profile local-model up -d ollama protometer_app   (OLLAMA_URL=http://ollama:11434)"
fi
