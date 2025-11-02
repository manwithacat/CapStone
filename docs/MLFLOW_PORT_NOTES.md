# MLflow Port Configuration

## Port Change: 5000 → 5001

**Issue**: macOS uses port 5000 for **AirPlay Receiver** (AirTunes service), which conflicts with MLflow's default port.

**Error**: When accessing `http://localhost:5000` in Safari, you would get:
```
403 Forbidden
Server: AirTunes/920.10.1
```

**Solution**: MLflow is now configured to use **port 5001** by default.

## Updated Configuration

### Scripts
- `scripts/mlflow_start.sh` - Default port changed to 5001
- Comment added explaining the macOS AirPlay conflict

### Makefile
- `make mlflow-start` - Uses port 5001
- `make mlflow-ui` - Opens http://localhost:5001
- Help text updated to show port 5001

### Access MLflow UI

**Browser**: http://localhost:5001

**Command**:
```bash
make mlflow-ui
# or manually:
open http://localhost:5001
```

## Using a Different Port

If port 5001 is also in use, you can specify a custom port:

```bash
# Start with custom port
./scripts/mlflow_start.sh 5002

# Check what's using a port
lsof -i :5001
```

## Disabling macOS AirPlay (Alternative)

If you prefer to use port 5000:

1. Open **System Settings**
2. Go to **General** → **AirDrop & Handoff**
3. Disable **AirPlay Receiver**
4. Revert MLflow to port 5000:
   ```bash
   ./scripts/mlflow_start.sh 5000
   ```

**Note**: This is not recommended as it disables a system feature.

## Related Documentation

- MLflow Quick Start: `docs/MLFLOW_QUICKSTART.md`
- MLflow Strategy: `docs/MLFLOW_IMPLEMENTATION_STRATEGY.md`
- Server Scripts: `scripts/mlflow_start.sh`, `scripts/mlflow_stop.sh`
