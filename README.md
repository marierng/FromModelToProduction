# From Model to Production – Fashion-MNIST Refund-Item Classifier

This repository shows how to take a computer-vision model from training to a production-ready service that runs nightly batch predictions.

| Item | Details |
|------|---------|
| Dataset | Fashion-MNIST (70 000 grayscale images of clothing; MIT licence) |
| Frameworks | TensorFlow 2.13, Flask 2, Gunicorn, GitHub Actions, Render |
| Test accuracy | About 91 % after 10 epochs (see output of `model.py`) |

---

## 1  Quick start

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/marierng/FromModelToProduction.git
cd FromModelToProduction
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
