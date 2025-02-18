# Contribution Guidelines

### Workflow

1. Create a feature branch:

   ```bash

   git checkout -b feature/{feature_name}

   ```


2. Rebase frequently with `develop`:

   ```bash

   git fetch origin

   git rebase origin/develop

   ```


3. Submit a pull request to merge into `develop`.

---

### Best Practices

- Write clear and concise commit messages.
- Ensure all tests pass before submitting.
- Use descriptive branch names.
