#!/usr/bin/env python3
"""
Release automation script for findingmodel package.

This script automates the release process including:
- Pre-flight checks (clean git, tests passing, etc.)
- Version management
- Changelog updates
- Building packages
- Git operations (merge, tag, push)
- PyPI publishing
- GitHub release creation
- Post-release cleanup

Usage:
    python scripts/release.py                    # Interactive mode
    python scripts/release.py --version 0.3.2   # Specify version
    python scripts/release.py --dry-run          # Preview actions
    python scripts/release.py --check-only       # Just run checks
    python scripts/release.py --yes              # Skip confirmations
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

from loguru import logger


class ReleaseError(Exception):
    """Custom exception for release process errors."""

    pass


class ReleaseManager:
    """Manages the complete release process for the findingmodel package."""

    def __init__(
        self, version: Optional[str] = None, dry_run: bool = False, yes: bool = False, check_only: bool = False
    ) -> None:
        self.version = version
        self.dry_run = dry_run
        self.yes = yes
        self.check_only = check_only
        self.project_root = Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.changelog_path = self.project_root / "CHANGELOG.md"
        self.dist_path = self.project_root / "dist"

        # Setup logging
        self._setup_logging()

        # Log startup info
        logger.info("Starting release process")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Target version: {version or 'TBD'}")
        logger.info(f"Dry run: {dry_run}")
        logger.info(f"Check only: {check_only}")
        logger.info(f"Auto-confirm: {yes}")

    def _setup_logging(self) -> None:
        """Configure loguru logging with file output."""
        # Remove default logger
        logger.remove()

        # Add console logger
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
        )

        # Add file logger
        if self.version:
            log_file = self.project_root / f"release-v{self.version}.log"
        else:
            log_file = self.project_root / f"release-{datetime.now():%Y%m%d-%H%M%S}.log"

        logger.add(
            log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG", rotation="1 MB"
        )

        self.log_file = log_file
        logger.info(f"Detailed logs: {log_file}")

    def run_command(
        self, cmd: str, check: bool = True, capture_output: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Execute a shell command with logging and optional dry-run."""
        logger.debug(f"Executing: {cmd}")

        # Commands that should run even in dry-run mode (read-only operations)
        readonly_commands = [
            "git branch --show-current",
            "git status --porcelain", 
            "git rev-list",
            "git tag -l",
            "git fetch origin"
        ]
        
        should_run_readonly = any(cmd.startswith(readonly_cmd) for readonly_cmd in readonly_commands)

        if self.dry_run and not should_run_readonly:
            logger.info(f"[DRY RUN] Would execute: {cmd}")
            # Return a mock successful result for dry run
            return subprocess.CompletedProcess(cmd, 0, "", "")

        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, cwd=self.project_root)

        if result.stdout and capture_output:
            logger.debug(f"stdout: {result.stdout.strip()}")
        if result.stderr and capture_output:
            logger.debug(f"stderr: {result.stderr.strip()}")

        if check and result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise ReleaseError(f"Command failed: {cmd}")

        return result

    def confirm(self, message: str) -> bool:
        """Get user confirmation unless --yes flag is used."""
        if self.yes:
            logger.info(f"Auto-confirming: {message}")
            return True

        print(f"\n🤔 {message}")
        response = input("Continue? [y/N]: ").strip().lower()
        confirmed = response == "y"
        logger.info(f"User confirmation for '{message}': {confirmed}")
        return confirmed

    def get_current_version(self) -> str:
        """Read current version from pyproject.toml."""
        logger.debug("Reading current version from pyproject.toml")

        if not self.pyproject_path.exists():
            raise ReleaseError("pyproject.toml not found")

        # Use uv to get version instead of parsing TOML directly
        result = self.run_command("uv version --quiet", check=False)
        if result.returncode == 0 and result.stdout.strip():
            current_version = str(result.stdout.strip())
        else:
            # Fallback: parse pyproject.toml manually
            with open(self.pyproject_path, "r") as f:
                content = f.read()

            # Simple regex to extract version
            version_match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if not version_match:
                raise ReleaseError("Version not found in pyproject.toml")
            current_version = str(version_match.group(1))

        logger.info(f"Current version: {current_version}")
        return current_version

    def validate_version(self, version: str) -> bool:
        """Validate version format (semantic versioning)."""
        logger.debug(f"Validating version format: {version}")

        # Basic semantic version pattern
        pattern = r"^\d+\.\d+\.\d+([.-]?(alpha|beta|rc)\d*)?$"
        if not re.match(pattern, version):
            logger.error(f"Invalid version format: {version}")
            return False

        logger.info(f"Version format valid: {version}")
        return True

    def check_git_status(self) -> None:
        """Ensure git working directory is clean."""
        logger.info("Checking git status...")

        result = self.run_command("git status --porcelain")
        if result.stdout.strip():
            logger.error("Git working directory is not clean:")
            logger.error(result.stdout)
            raise ReleaseError("Please commit or stash your changes before releasing")

        logger.success("✅ Git working directory is clean")

    def check_branch(self) -> None:
        """Ensure we're on dev branch and up to date."""
        logger.info("Checking current branch...")

        # Check current branch
        result = self.run_command("git branch --show-current")
        current_branch = result.stdout.strip()

        if current_branch != "dev":
            raise ReleaseError(f"Must be on 'dev' branch, currently on '{current_branch}'")

        logger.success("✅ On dev branch")

        # Check if up to date with remote
        logger.info("Checking if branch is up to date with origin...")
        self.run_command("git fetch origin")

        result = self.run_command("git rev-list HEAD...origin/dev --count")
        behind_count = int(result.stdout.strip())

        if behind_count > 0:
            raise ReleaseError(f"Branch is {behind_count} commits behind origin/dev. Please pull latest changes.")

        logger.success("✅ Branch is up to date with origin")

    def check_existing_tag(self, version: str) -> None:
        """Check if version tag already exists."""
        logger.info(f"Checking if tag v{version} already exists...")

        result = self.run_command(f"git tag -l v{version}", check=False)
        if result.stdout.strip():
            raise ReleaseError(f"Tag v{version} already exists")

        logger.success(f"✅ Tag v{version} does not exist")

    def run_tests(self) -> None:
        """Run the test suite."""
        logger.info("Running test suite...")

        # Run regular tests
        logger.info("Running regular tests (task test)...")
        result = self.run_command("task test", capture_output=False)
        if result.returncode != 0:
            raise ReleaseError("Regular tests failed")

        logger.success("✅ Regular tests passed")

        # Ask about full tests
        if not self.yes and not self.confirm("Run full test suite including API tests? (recommended)"):
            logger.warning("Skipping full test suite")
            return

        logger.info("Running full test suite (task test-full)...")
        result = self.run_command("task test-full", capture_output=False, check=False)

        if result.returncode != 0:
            logger.warning("⚠️ Full test suite had failures")
            if not self.confirm("Continue with release despite test failures?"):
                raise ReleaseError("Full test suite failed")
        else:
            logger.success("✅ Full test suite passed")

    def run_checks(self) -> None:
        """Run code quality checks."""
        logger.info("Running code quality checks...")

        result = self.run_command("task check", capture_output=False)
        if result.returncode != 0:
            raise ReleaseError("Code quality checks failed")

        logger.success("✅ Code quality checks passed")

    def update_version(self) -> None:
        """Update version in pyproject.toml using uv."""
        if not self.version:
            current = self.get_current_version()
            print(f"\nCurrent version: {current}")
            self.version = input("Enter new version: ").strip()

        if not self.validate_version(self.version):
            raise ReleaseError(f"Invalid version: {self.version}")

        self.check_existing_tag(self.version)

        logger.info(f"Updating version to {self.version}")
        self.run_command(f"uv version {self.version}")

        # Update log file name now that we have a version
        if not hasattr(self, "log_file") or "release-v" not in str(self.log_file):
            self._setup_logging()

        logger.success(f"✅ Version updated to {self.version}")

    def update_changelog(self) -> None:
        """Update CHANGELOG.md with release date."""
        logger.info("Updating CHANGELOG.md...")

        if not self.changelog_path.exists():
            logger.warning("CHANGELOG.md not found, skipping changelog update")
            return

        with open(self.changelog_path, "r") as f:
            content = f.read()

        # Look for version entry
        if self.version is None:
            raise ReleaseError("Version not set")
        version_pattern = rf"^## \[?{re.escape(self.version)}\]?"
        if not re.search(version_pattern, content, re.MULTILINE):
            logger.warning(f"Version {self.version} not found in CHANGELOG.md")
            if not self.confirm("Continue without updating changelog?"):
                raise ReleaseError("Changelog not updated")
            return

        # Update date
        if self.version is None:
            raise ReleaseError("Version not set")
        today = datetime.now().strftime("%Y-%m-%d")
        updated_content = re.sub(
            rf"^## \[?{re.escape(self.version)}\]?.*$", f"## [{self.version}] - {today}", content, flags=re.MULTILINE
        )

        if not self.dry_run:
            with open(self.changelog_path, "w") as f:
                f.write(updated_content)

        logger.success("✅ CHANGELOG.md updated with release date")

    def clean_dist(self) -> None:
        """Clean the dist directory of old builds."""
        logger.info("Cleaning dist directory...")

        if self.dist_path.exists():
            if not self.dry_run:
                shutil.rmtree(self.dist_path)
            logger.info("Removed existing dist directory")

        logger.success("✅ Dist directory cleaned")

    def build_packages(self) -> List[Path]:
        """Build wheel and source distribution packages."""
        logger.info("Building packages...")

        self.run_command("uv build", capture_output=False)

        if self.dry_run:
            return []

        # Check built files
        wheel_pattern = f"findingmodel-{self.version}-py3-none-any.whl"
        sdist_pattern = f"findingmodel-{self.version}.tar.gz"

        wheel_path = self.dist_path / wheel_pattern
        sdist_path = self.dist_path / sdist_pattern

        built_files = []
        for file_path, name in [(wheel_path, "wheel"), (sdist_path, "source distribution")]:
            if not file_path.exists():
                raise ReleaseError(f"Expected {name} not found: {file_path}")

            size = file_path.stat().st_size
            logger.info(f"Built {name}: {file_path.name} ({size:,} bytes)")

            # Calculate checksum
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                sha256.update(f.read())
            logger.debug(f"SHA256: {sha256.hexdigest()}")

            built_files.append(file_path)

        logger.success("✅ Packages built successfully")
        return built_files

    def commit_changes(self) -> None:
        """Commit version and changelog changes."""
        logger.info("Committing version and changelog changes...")

        # Check if there are any changes to commit
        status_result = self.run_command("git status --porcelain")
        if not status_result.stdout.strip():
            logger.info("No changes to commit - version and changelog already up to date")
            logger.success("✅ Repository already clean, skipping commit")
            return

        self.run_command("git add pyproject.toml")
        if self.changelog_path.exists():
            self.run_command("git add CHANGELOG.md")

        commit_msg = f"Release v{self.version}"
        self.run_command(f'git commit -m "{commit_msg}"')

        logger.success(f"✅ Changes committed: {commit_msg}")

    def merge_to_main(self) -> None:
        """Merge dev branch to main."""
        logger.info("Merging dev to main...")

        if not self.confirm("Ready to merge dev to main?"):
            raise ReleaseError("User aborted merge to main")

        # Switch to main
        self.run_command("git checkout main")

        # Pull latest main
        self.run_command("git pull origin main")

        # Merge dev with no-ff to maintain history
        self.run_command(f"git merge --no-ff dev -m 'Merge dev for release v{self.version}'")

        logger.success("✅ Merged dev to main")

    def create_tag(self) -> None:
        """Create and push git tag."""
        logger.info(f"Creating tag v{self.version}...")

        tag_msg = f"Release v{self.version}"
        self.run_command(f'git tag -a v{self.version} -m "{tag_msg}"')

        logger.info("Pushing main branch and tags...")
        self.run_command("git push origin main")
        self.run_command("git push origin --tags")

        logger.success(f"✅ Tag v{self.version} created and pushed")

    def publish_pypi(self) -> None:
        """Publish packages to PyPI using uv publish."""
        logger.info("Publishing to PyPI...")

        if not self.confirm(f"Ready to publish v{self.version} to PyPI?"):
            logger.warning("PyPI publishing skipped by user")
            return

        self.run_command("uv publish", capture_output=False)

        pypi_url = f"https://pypi.org/project/findingmodel/{self.version}/"
        logger.success(f"✅ Published to PyPI: {pypi_url}")

    def create_github_release(self, built_files: List[Path]) -> None:
        """Create GitHub release with artifacts."""
        logger.info("Creating GitHub release...")

        if not self.confirm("Create GitHub release?"):
            logger.warning("GitHub release skipped by user")
            return

        # Create release with auto-generated notes
        release_cmd = f"gh release create v{self.version} --title 'Release v{self.version}' --generate-notes"

        # Add built files as assets
        for file_path in built_files:
            release_cmd += f' "{file_path}"'

        self.run_command(release_cmd)

        github_url = f"https://github.com/talkasab/findingmodel/releases/tag/v{self.version}"
        logger.success(f"✅ GitHub release created: {github_url}")

    def post_release_cleanup(self) -> None:
        """Switch back to dev branch and prepare for next development cycle."""
        logger.info("Performing post-release cleanup...")

        # Switch back to dev
        self.run_command("git checkout dev")

        # Merge main back to dev to get the merge commit
        self.run_command("git merge main")

        # Push updated dev branch
        self.run_command("git push origin dev")

        logger.success("✅ Post-release cleanup completed")

    def print_summary(self, built_files: List[Path]) -> None:
        """Print release summary."""
        print("\n" + "=" * 60)
        print(f"🎉 Release v{self.version} completed successfully!")
        print("=" * 60)

        print("\n📦 Built packages:")
        for file_path in built_files:
            print(f"   • {file_path.name}")

        print("\n🔗 URLs:")
        print(f"   • PyPI: https://pypi.org/project/findingmodel/{self.version}/")
        print(f"   • GitHub: https://github.com/talkasab/findingmodel/releases/tag/v{self.version}")

        print("\n📋 What was done:")
        print(f"   • Version updated to {self.version}")
        print("   • CHANGELOG.md updated with release date")
        print("   • Changes committed to git")
        print("   • Dev merged to main")
        print(f"   • Tag v{self.version} created and pushed")
        print("   • Packages published to PyPI")
        print("   • GitHub release created with artifacts")
        print("   • Switched back to dev branch")

        print(f"\n📄 Log file: {self.log_file}")
        print("\n✨ Great job! The release is complete.")

    def handle_error(self, step_name: str, error: Exception) -> None:
        """Handle errors and provide recovery instructions."""
        logger.error(f"❌ Failed at step: {step_name}")
        logger.error(f"Error: {error}")

        recovery_instructions = {
            "check_git_status": "Fix git status issues and retry",
            "check_branch": "Switch to dev branch and ensure it's up to date",
            "run_tests": "Fix failing tests and retry",
            "merge_to_main": "Run 'git merge --abort' then 'git checkout dev'",
            "create_tag": "Tag not created, no cleanup needed",
            "publish_pypi": "Package built but not published, check dist/ folder",
            "create_github_release": "Package is on PyPI but no GitHub release, create manually",
        }

        if step_name in recovery_instructions:
            logger.info(f"💡 Recovery suggestion: {recovery_instructions[step_name]}")

        logger.info(f"📄 Full log available in: {self.log_file}")

        # If we're on main branch and something failed, suggest switching back
        try:
            result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
            if result.stdout.strip() == "main":
                logger.info("💡 You're on main branch. Consider running 'git checkout dev' to return to development.")
        except Exception:
            pass

    def _get_release_steps(self) -> list[tuple[str, str, Callable[..., Any]]]:
        """Get the list of release steps."""
        return [
            ("check_git_status", "Checking git status", self.check_git_status),
            ("check_branch", "Checking branch", self.check_branch),
            ("run_tests", "Running tests", self.run_tests),
            ("run_checks", "Running code quality checks", self.run_checks),
            ("update_version", "Updating version", self.update_version),
            ("update_changelog", "Updating changelog", self.update_changelog),
            ("clean_dist", "Cleaning dist directory", self.clean_dist),
            ("build_packages", "Building packages", self.build_packages),
            ("commit_changes", "Committing changes", self.commit_changes),
            ("merge_to_main", "Merging to main", self.merge_to_main),
            ("create_tag", "Creating git tag", self.create_tag),
            ("publish_pypi", "Publishing to PyPI", self.publish_pypi),
            ("create_github_release", "Creating GitHub release", self.create_github_release),
            ("post_release_cleanup", "Post-release cleanup", self.post_release_cleanup),
        ]

    def _should_skip_step(self, step_id: str) -> bool:
        """Check if step should be skipped in check-only mode."""
        skip_steps = {
            "update_version",
            "update_changelog",
            "clean_dist",
            "build_packages",
            "commit_changes",
            "merge_to_main",
            "create_tag",
            "publish_pypi",
            "create_github_release",
            "post_release_cleanup",
        }
        return self.check_only and step_id in skip_steps

    def _execute_step(self, step_id: str, step_func: Callable[..., Any], built_files: List[Path]) -> List[Path]:
        """Execute a single release step."""
        if step_id == "build_packages":
            result: Any = step_func()
            if isinstance(result, list):
                return result
            return built_files
        elif step_id == "create_github_release":
            step_func(built_files)
            return built_files
        else:
            step_func()
            return built_files

    def release(self) -> None:
        """Execute the complete release process."""
        if self.dry_run:
            logger.info("🏃 Starting DRY RUN - no changes will be made")
        else:
            logger.info("🚀 Starting RELEASE process")

        steps = self._get_release_steps()
        built_files: List[Path] = []

        try:
            for step_id, step_name, step_func in steps:
                if self._should_skip_step(step_id):
                    logger.info(f"⏭️ Skipping {step_name} (check-only mode)")
                    continue

                print(f"\n{'=' * 60}")
                print(f"📋 {step_name}...")
                print("=" * 60)

                try:
                    built_files = self._execute_step(step_id, step_func, built_files)
                    logger.success(f"✅ {step_name} completed")
                except Exception as e:
                    self.handle_error(step_id, e)
                    raise

        except KeyboardInterrupt:
            logger.warning("Release process interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Release process failed: {e}")
            sys.exit(1)

        if not self.check_only and not self.dry_run:
            self.print_summary(built_files)
        elif self.check_only:
            logger.success("✅ All pre-release checks passed!")
        elif self.dry_run:
            logger.success("🏃 Dry run completed successfully!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automate findingmodel package releases")
    parser.add_argument("--version", help="Version to release (e.g., 0.3.2)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--check-only", action="store_true", help="Only run pre-flight checks")

    args = parser.parse_args()

    try:
        manager = ReleaseManager(version=args.version, dry_run=args.dry_run, yes=args.yes, check_only=args.check_only)
        manager.release()

    except KeyboardInterrupt:
        print("\n\nRelease process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Release process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
