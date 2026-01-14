# Pre-push safety check - Detect conflicts before pushing
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess

import oteam


DIR_RUNTIME = ".oteam/runtime"


def check_pre_push(root: Path, agent_name: str) -> Tuple[bool, List[str]]:
    """Run pre-push safety checks.

    Args:
        root: Workspace root path
        agent_name: Agent name

    Returns:
        Tuple of (all_passed, list_of_issues)
    """
    issues = []

    repo = root / oteam.DIR_AGENTS / agent_name / "proj"
    if not repo.exists():
        issues.append(f"Agent repo not found: {repo}")
        return False, issues

    if not _is_clean(repo):
        issues.append("Working tree is not clean. Commit or stash changes.")

    if not _has_upstream(repo):
        issues.append(
            "No upstream branch configured. Set with: git branch --set-upstream-to"
        )

    local_branch = _get_local_branch(repo)
    remote_branch = _get_remote_branch(repo, local_branch)

    if local_branch and remote_branch:
        behind, ahead = _get_behind_ahead(repo, local_branch, remote_branch)
        if behind > 0:
            issues.append(
                f"Local branch is {behind} commits behind upstream. Pull first."
            )
        if ahead > 0:
            issues.append(f"Local branch is {ahead} commits ahead. Push will succeed.")

    if _has_untracked_files(repo):
        issues.append(
            "Untracked files exist. These won't be pushed but should be committed."
        )

    ticket_id = _extract_ticket_id(local_branch)
    if ticket_id:
        ticket = oteam._find_ticket_by_id(root, ticket_id)
        if ticket and ticket.get("assignee") != agent_name:
            issues.append(
                f"Ticket T{ticket_id} is assigned to {ticket.get('assignee')}, not {agent_name}"
            )

    return len(issues) == 0, issues


def _is_clean(repo: Path) -> bool:
    """Check if git working tree is clean."""
    try:
        cp = oteam.run_cmd(["git", "status", "--porcelain"], cwd=repo, capture=True)
        return not cp.stdout.strip()
    except Exception:
        return False


def _has_upstream(repo: Path) -> bool:
    """Check if branch has upstream configured."""
    try:
        cp = oteam.run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "@{upstream}"], cwd=repo, capture=True
        )
        return bool(cp.stdout.strip())
    except Exception:
        return False


def _get_local_branch(repo: Path) -> Optional[str]:
    """Get current local branch name."""
    try:
        cp = oteam.run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture=True
        )
        return cp.stdout.strip() or None
    except Exception:
        return None


def _get_remote_branch(repo: Path, local_branch: str) -> Optional[str]:
    """Get remote branch name for local branch."""
    if not local_branch:
        return None
    try:
        cp = oteam.run_cmd(
            ["git", "rev-parse", "--abbrev-ref", f"{local_branch}@{{upstream}}"],
            cwd=repo,
            capture=True,
        )
        return cp.stdout.strip() or None
    except Exception:
        return None
    try:
        cp = oteam.run_cmd(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                f"{local_branch}@{{{'upstream'}}}".format(),
            ],
            cwd=repo,
            capture=True,
        )
        return cp.stdout.strip() or None
    except Exception:
        return None


def _get_behind_ahead(repo: Path, local: str, remote: str) -> Tuple[int, int]:
    """Get behind/ahead counts between local and remote branch."""
    try:
        cp = oteam.run_cmd(
            ["git", "rev-list", "--left-right", f"{local}...{remote}", "--count"],
            cwd=repo,
            capture=True,
        )
        parts = cp.stdout.strip().split()
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def _has_untracked_files(repo: Path) -> bool:
    """Check if there are untracked files."""
    try:
        cp = oteam.run_cmd(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo,
            capture=True,
        )
        return bool(cp.stdout.strip())
    except Exception:
        return False


def _extract_ticket_id(branch_name: Optional[str]) -> Optional[str]:
    """Extract ticket ID from branch name like 'agent/dev1/T3'."""
    if not branch_name:
        return None
    if branch_name.startswith("agent/"):
        parts = branch_name.split("/")
        if len(parts) >= 3:
            return parts[-1]
    return None


def run_pre_push_check(workdir: str, agent: str) -> bool:
    """CLI entry point for pre-push check."""
    root = oteam.find_project_root(Path(workdir))
    if not root:
        print("Error: Not an oteam workspace")
        return False

    all_passed, issues = check_pre_push(root, agent)

    if all_passed:
        print("✓ Pre-push check passed")
        print("  Ready to push!")
        return True
    else:
        print("✗ Pre-push check failed")
        for issue in issues:
            print(f"  - {issue}")
        return False


def check_conflicts_with_main(repo: Path) -> List[str]:
    """Check for potential conflicts with main branch."""
    conflicts = []

    try:
        cp = oteam.run_cmd(
            ["git", "log", "--oneline", "main..HEAD"],
            cwd=repo,
            capture=True,
        )
        local_commits = [l for l in cp.stdout.strip().split("\n") if l]

        cp = oteam.run_cmd(
            ["git", "log", "--oneline", "HEAD..main"],
            cwd=repo,
            capture=True,
        )
        remote_commits = [l for l in cp.stdout.strip().split("\n") if l]

        if remote_commits and local_commits:
            conflicts.append(
                f"Main has {len(remote_commits)} commits not in local branch"
            )
            conflicts.append(f"Local has {len(local_commits)} commits not in main")

    except Exception:
        pass

    return conflicts
