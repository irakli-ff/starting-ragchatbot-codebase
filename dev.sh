#!/bin/bash

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Running Code Quality Checks ===${NC}"

if [ "$1" == "format" ]; then
    echo -e "${YELLOW}Formatting code with Black...${NC}"
    uv run black backend/ --exclude chroma_db
    echo -e "${YELLOW}Sorting imports with isort...${NC}"
    uv run isort backend/ --skip chroma_db
    echo -e "${GREEN}✓ Code formatted successfully${NC}"
    exit 0
fi

if [ "$1" == "check" ] || [ -z "$1" ]; then
    echo -e "${YELLOW}Checking code format with Black...${NC}"
    uv run black backend/ --check --exclude chroma_db || {
        echo -e "${RED}✗ Black formatting check failed. Run './dev.sh format' to fix.${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Black check passed${NC}"
    
    echo -e "${YELLOW}Checking import sorting with isort...${NC}"
    uv run isort backend/ --check-only --skip chroma_db || {
        echo -e "${RED}✗ Import sorting check failed. Run './dev.sh format' to fix.${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ isort check passed${NC}"
    
    echo -e "${YELLOW}Running flake8 linter...${NC}"
    uv run flake8 backend/ || {
        echo -e "${RED}✗ Flake8 linting failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Flake8 check passed${NC}"
    
    echo -e "${YELLOW}Running mypy type checker...${NC}"
    uv run mypy backend/ || {
        echo -e "${RED}✗ Mypy type checking failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Mypy check passed${NC}"
fi

if [ "$1" == "test" ]; then
    echo -e "${YELLOW}Running tests with pytest...${NC}"
    uv run pytest backend/tests/ -v || {
        echo -e "${RED}✗ Tests failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ All tests passed${NC}"
fi

if [ "$1" == "all" ]; then
    ./dev.sh check
    ./dev.sh test
fi

if [ "$1" == "pre-commit" ]; then
    echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
    uv run pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

if [ "$1" == "help" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check       Run all code quality checks (default)"
    echo "  format      Format code with Black and isort"
    echo "  test        Run test suite with pytest"
    echo "  all         Run checks and tests"
    echo "  pre-commit  Install pre-commit hooks"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./dev.sh          # Run quality checks"
    echo "  ./dev.sh format   # Format code"
    echo "  ./dev.sh all      # Run everything"
fi

if [ -z "$1" ] || [ "$1" == "check" ]; then
    echo -e "${GREEN}=== All quality checks passed! ===${NC}"
fi