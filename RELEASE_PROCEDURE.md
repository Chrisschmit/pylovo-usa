# Release Procedure

This release procedure outlines the steps for managing releases in the GitHub environment.<br>
These symbols help with orientation:
- 🐙 GitHub
- 💠 git (Bash)
- 📝 File
- 💻 Command Line (CMD)


## Version Numbers

This software follows the [Semantic Versioning (SemVer)](https://semver.org/).<br>
It always has the format `MAJOR.MINOR.PATCH`, e.g. `1.5.0`.

The data follows the [Calendar Versioning (CalVer)](https://calver.org/).<br>
It always has the format `YYYY-MM-DD`, e.g. `1992-11-07`.


## GitHub Release

### 1. Update the `CHANGELOG.md`
- 📝 **File**: Open the CHANGELOG.md file and add a new entry under the `[Unreleased]` section.
- 💠 **Commit**: Commit your changes to the changelog, noting all new features, changes, and fixes.

### 2. Create a `Draft GitHub Release` Issue
- 🐙 **Template**: Use the `📝Release_Checklist` template for the issue.
- 🐙 **Issue**: Create a new issue in the repository with the title `Release - Minor Version - 0.1.0`.
- 🐙 **Description**: Fill in the details of the release, including the name, Git tag, release manager, and date.
- 🐙 **Workflow Checklist**: Check off the steps in the workflow checklist for the release.

### 3. Create a Release Branch
- 💠 **Branching**: Create a release branch from develop:
    ```bash
    git checkout dev
    git pull
    git checkout -b release-1.5.0
    ```
- 💠 **Push**: Push the release branch to GitHub:
    ```bash
    git push --set-upstream origin release-1.5.0
    ```

### 4. Finalize and Merge
- 🐙 **Merge Request**: In GitHub, open a merge request (MR) from `release-1.5.0` into `main`.
- 🐙 **Review**: Assign reviewers to the MR and ensure all tests pass.
- 🐙 **Merge**: Once approved, merge the MR into main and delete the release branch.

### 5. Tag the Release
- 💠 **Checkout** main: Ensure you’re on the main branch.
    ```bash
    git checkout main
    git pull
    ```
- 💠 **Tag**: Tag the new release in GitHub:
    ```bash
    git tag -a v1.5.0 -m "Release 1.5.0"
    git push origin v1.5.0
    ```

### 6. Create a GitHub Release (Optional)
- 🐙 **GitHub Release Page**: Go to the GitHub project’s Releases section and create a new release linked to the v1.5.0 tag.
- 📝 **Release Notes**: Add release notes using information from the changelog.

### 7. Update the Documentation
- 📝 **Documentation**: Update the documentation to reflect the new release version.
- 💻 **Build**: Build the documentation to ensure it’s up to date.
- 💻 **Deploy**: Deploy the documentation to the appropriate location.
- 💻 **Update**: Update any version references in the documentation.
- 💻 **Commit**: Commit the documentation changes.
- 💠 **Push**: Push the documentation changes to the repository.
- 🐙 **Merge**: Merge the documentation changes into the main branch.
- 🐙 **Delete Branch**: Delete the release branch after merging.

### 8. Merge Back into `develop`
- 💠 **Branch**: Create an MR from `main` into `develop` to merge the release changes back into the development branch.
```bash
git checkout develop
git pull
git merge main
git push
```

### Important Notes
- **Versioning**: Always increment the version correctly using `bump2version` before creating the final release.
- **Publishing Reminder**: Ensure your PyPI credentials are correctly set up in GitHub CI/CD or local `.pypirc` configuration for seamless uploads.
- **Final Check**: If issues arise post-release, refer to the [GitHub CI/CD guide](https://docs.GitHub.com/ee/development/cicd/) and [PyPI documentation](https://packaging.python.org/en/latest/) for troubleshooting.
