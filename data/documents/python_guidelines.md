# Python Development Guidelines

## Code Style

### PEP 8 Compliance
Follow Python's official style guide:
- Use 4 spaces for indentation (not tabs)
- Limit lines to 79 characters (99 for modern projects)
- Use snake_case for functions and variables
- Use PascalCase for class names
- Use UPPER_CASE for constants

### Type Hints
Always use type hints for function signatures:
```python
def calculate_total(items: list[float], tax_rate: float = 0.1) -> float:
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)
```

### Docstrings
Use Google-style docstrings:
```python
def fetch_user(user_id: int) -> dict:
    """Fetch user data from the database.
    
    Args:
        user_id: The unique identifier for the user.
        
    Returns:
        Dictionary containing user profile data.
        
    Raises:
        UserNotFoundError: If user doesn't exist.
    """
```

## Project Structure

### Standard Layout
```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── main.py
│       └── utils/
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Testing

### Unit Tests
- Test one thing per test function
- Use descriptive test names
- Arrange-Act-Assert pattern
- Aim for 80%+ coverage

### Fixtures
Use pytest fixtures for setup:
```python
@pytest.fixture
def sample_user():
    return {"id": 1, "name": "Test User"}
```

## Error Handling

### Exception Hierarchy
- Use specific exceptions over generic ones
- Create custom exceptions for domain errors
- Always log exceptions before re-raising

### Context Managers
Use for resource management:
```python
with open("data.txt", "r") as f:
    content = f.read()
```

## Performance

### Profiling
Use cProfile for CPU profiling:
```bash
python -m cProfile -s cumulative my_script.py
```

### Memory
Use memory_profiler for memory analysis:
```python
@profile
def memory_intensive_function():
    pass
```
