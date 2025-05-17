# Contribution Guidelines

Thank you for considering contributing to PawPrint! We appreciate your interest in helping us improve our project. Please follow these guidelines to ensure a smooth contribution process.

## How to Contribute

Please read [Contributing to a project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) for a general overview of the contribution process. You should follow these steps:
1. **Fork the repository**: Fork the repository and clone it to your local machine.

2. **Create a new branch**: Create a new branch for your changes:
    ```bash
    git checkout -b feature/your-feature-name
    ```

3. **Make your changes**: Make the necessary changes to the codebase. You should follow the project's coding style and conventions:
    - Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. You can use [Pylint](https://pypi.org/project/pylint/) to check your code for style issues. You can use the Pylint integration in your IDE (e.g., VSCode or PyCharm) or run it from the command line:
        ```bash
        pylint your_module.py
        ```
    - Use [Black](https://black.readthedocs.io/en/stable/) to format your code before committing. You can run it with:
        ```bash
        black .
        ```
    - Write docstrings for your files, functions and classes. Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.
    - Write simple tests for your code. We use [pytest](https://docs.pytest.org/en/stable/) for testing. You can run the tests with:
        ```bash
        pytest
        ```
4. **Commit your changes**: Commit your changes with a descriptive commit message following the [Conventional Commits](https://www.conventionalcommits.org/) specification. For example:
    ```bash
    git commit -m "feat: add new feature"
    ```
    You can use [commit message editor](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor) for VSCode to help you write commit messages. 
5. **Push to your fork**: Push your changes to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
6. **Create a pull request**: Go to the original repository and create a pull request from your branch. Provide a clear description of your changes and why they are needed.
