# Reviewer matcher 🤺
Automatic tool to support the assignment of reviewers to project proposals
### Objective
- Project: Develop a system to facilitate the assignment of expert reviewers to project proposals.
- Current step: Obtain a set of potential reviewers such that their expertise aligns closely with the project's objectives and methods.

### Approach
- Leverage NLP techniques to implement a multi-stage process involving data retrieval, content enrichment, and similarity computation.
- Obtain, as a result, a score and rank for each potential expert-project assignment.

## Preliminary code structure

In the image, we provide an overview of the core modules of the tool, highlighting its fundamental components and their primary functions. Each module represents a key area of functionality, illustrating how different parts of the tool work together to deliver a cohesive experience. This breakdown serves as an introduction to the essential building blocks of the tool, helping users understand its architecture and operational flow.

![image](https://github.com/user-attachments/assets/e47e2ad0-8946-4ad9-84ee-ecba3c8783f4)

## Data

The data shown for demonstartion is artificially generated to demonstrate the tool’s capabilities, as the actual training data is confidential and cannot be shared publicly.

https://docs.google.com/spreadsheets/d/1lUBxxTinGEsp1tmXDTsw29en6hKfYR798QdEFgQAPOU/edit?gid=0#gid=0

Instead of a single main branch, we use two branches to record the history of the project:

- `develop`: development and default branch for new features and bug fixes.
- `main`: production branch is used to deploy the server components to the `production` environment.

Please, follow the 📗 [Contribution guidelines](docs/contribute.md) in order to participate in this project.

## 📚 Documentation on Classes

For a detailed breakdown of each class and its methods, refer to our [Class Documentation](docs/index.md).

### 🛟 A note on language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).
