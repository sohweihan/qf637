# Project Setup and Git Workflow Guide

This guide explains how to clone the repository, create your own branch, make changes, and push your branch for review.

Please do **not** push directly to `master` and do **not** merge your own branch into `master`.

---

## 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository-url>
```

Example:

```bash
git clone https://github.com/username/repository-name.git
```

Then move into the project folder:

```bash
cd repository-name
```

---

## 2. Check Your Current Branch

To see which branch you are currently on:

```bash
git branch
```

The branch with the `*` beside it is your current branch.

Example:

```bash
* master
```

---

## 3. Pull the Latest Version of `master`

Before creating your own branch, make sure your local `master` branch is updated:

```bash
git checkout master
git pull origin master
```

Or, if your Git version uses `switch`:

```bash
git switch master
git pull origin master
```

---

## 4. Create a New Branch

Create a new branch for your own work:

```bash
git checkout -b <your-branch-name>
```

Example:

```bash
git checkout -b feature/add-login-page
```

You can also use:

```bash
git switch -c <your-branch-name>
```

Example:

```bash
git switch -c feature/add-login-page
```

Recommended branch naming examples:

```bash
feature/your-feature-name
fix/your-bug-fix
update/your-update-name
```

Examples:

```bash
feature/add-dashboard
fix/login-error
update/readme-instructions
```

---

## 5. Switch Between Branches

To switch to an existing branch:

```bash
git checkout <branch-name>
```

Example:

```bash
git checkout master
```

Or using the newer command:

```bash
git switch <branch-name>
```

Example:

```bash
git switch feature/add-login-page
```

---

## 6. Check File Changes

To see which files have been changed:

```bash
git status
```

This shows:
- Files that have been modified
- Files that are not yet tracked by Git
- Files that are ready to be committed

---

## 7. Add Your Changes

To add all changed files:

```bash
git add .
```

To add a specific file only:

```bash
git add <file-name>
```

Example:

```bash
git add README.md
```

---

## 8. Commit Your Changes

Commit your changes with a clear message:

```bash
git commit -m "Describe your changes"
```

Example:

```bash
git commit -m "Add login page layout"
```

Good commit messages should briefly explain what was changed.

Examples:

```bash
git commit -m "Fix navbar layout issue"
git commit -m "Add data cleaning script"
git commit -m "Update project documentation"
```

---

## 9. Push Your Branch

Push your branch to the remote repository:

```bash
git push origin <your-branch-name>
```

Example:

```bash
git push origin feature/add-login-page
```

After pushing, your branch will be available on GitHub/GitLab.

---

## 10. Create a Pull Request

After pushing your branch:

1. Go to the repository on GitHub/GitLab.
2. Open a Pull Request or Merge Request.
3. Select your branch as the source branch.
4. Select `master` as the target branch.
5. Add a clear title and description of your changes.
6. Submit the request for review.

Do **not** merge your branch into `master` yourself.

The repository maintainer will review your changes and merge the Pull Request if approved.

---

## 11. Updating Your Branch with the Latest `master`

If `master` has been updated and you need the latest changes, you can update your own branch.

First, switch to `master` and pull the latest changes:

```bash
git checkout master
git pull origin master
```

Then switch back to your own branch:

```bash
git checkout <your-branch-name>
```

Merge the latest `master` into your branch:

```bash
git merge master
```

Example:

```bash
git checkout master
git pull origin master
git checkout feature/add-login-page
git merge master
```

If there are conflicts, resolve them before committing and pushing again.

Important: This only updates your own branch with the latest `master`.  
You should still **not** merge your branch into `master` yourself.

---

## Common Git Commands Summary

```bash
# Clone repository
git clone <repository-url>

# Move into project folder
cd <repository-name>

# Check current branch
git branch

# Switch to master
git checkout master

# Pull latest master
git pull origin master

# Create and switch to a new branch
git checkout -b <branch-name>

# Switch to an existing branch
git checkout <branch-name>

# Check file changes
git status

# Add all changes
git add .

# Add a specific file
git add <file-name>

# Commit changes
git commit -m "Your commit message"

# Push your branch
git push origin <branch-name>
```

---

## Important Rules

Do **not** push directly to `master`.

Do **not** merge your own branch into `master`.

Always create a new branch for your own work.

Use clear and meaningful commit messages.

Push your branch and create a Pull Request for review.

The maintainer will handle the final merge into `master`.

---

## Typical Workflow

A typical workflow should look like this:

```bash
# 1. Clone the repository
git clone <repository-url>

# 2. Move into the project folder
cd <repository-name>

# 3. Switch to master
git checkout master

# 4. Pull the latest master
git pull origin master

# 5. Create your own branch
git checkout -b feature/my-new-feature

# 6. Make your changes in the files

# 7. Check changed files
git status

# 8. Add changes
git add .

# 9. Commit changes
git commit -m "Add my new feature"

# 10. Push your branch
git push origin feature/my-new-feature
```

After this, create a Pull Request on GitHub/GitLab.

Do **not** merge your own branch into `master`.
