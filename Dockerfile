FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backtesting service
COPY backtesting/ ./backtesting/

# Copy decision-engine (required for rule imports)
# In production, this should be a pip-installable package
COPY decision_engine/ ./decision_engine/

# Create non-root user
RUN useradd -m appuser
USER appuser

# Default entrypoint
ENTRYPOINT ["python", "-m", "backtesting.main"]
