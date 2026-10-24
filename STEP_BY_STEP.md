# Step-by-step guide (Windows + VS Code)

Do the steps in order. Every command goes in the **VS Code terminal**
(menu **Terminal → New Terminal**). Type or paste one line at a time and press Enter.

---

## Part 1 — Install the tools (one time)

1. **VS Code** — https://code.visualstudio.com/
2. **Git for Windows** — https://git-scm.com/download/win
   Keep all the default options in the installer.
3. **Python 3.12** — https://www.python.org/downloads/ (pick a 3.12.x version)
   On the first installer screen, tick **"Add python.exe to PATH"**.
4. **Node.js** (the "LTS" version, 22 or newer) — https://nodejs.org/
   (only needed to preview the page on your own computer)
5. A **GitHub account** — https://github.com/signup

Close and reopen VS Code after installing, then check each tool in the terminal:

```
git --version
py -3.12 --version
node --version
```

Each should print a version number. If one says *"not recognized"*, that tool is not
installed (or VS Code needs restarting).

---

## Part 2 — Open the project

1. Extract the project zip (right-click → **Extract All**) into a folder you can find,
   for example `Documents`.
2. In VS Code: **File → Open Folder…** → pick the extracted
   `FL-XAI-Intrusion-Detection-IIoT` folder (the one that contains `README.md`).
3. If VS Code asks *"Do you trust the authors?"*, click **Yes**.

---

## Part 3 — Python environment and dataset

1. Create and activate a Python environment:

   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

   The terminal line should now start with `(.venv)`.
   If you get a red error about *"running scripts is disabled"*, run this once, then
   run the activate line again:

   ```
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

2. Install the packages (takes a few minutes):

   ```
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. The dataset is **already included** as `data/ML-EdgeIIoT-dataset.csv.gz`
   (12.9 MB, compressed from the 78.4 MB Kaggle CSV). Nothing to do.

   *Only if you ever replace it:* put a new `ML-EdgeIIoT-dataset.csv` in the `data`
   folder and run `python scripts/compress_dataset.py` (it must end with a line
   starting with **OK**).

4. *(Optional but recommended)* run the tests to check everything works:

   ```
   python -m pytest
   ```

   It should end with `passed` and no `failed`.

---

## Part 4 — Create the GitHub repository

1. Go to https://github.com/new
2. **Repository name:** for example `flxai-iiot-reproduction`
3. Choose **Public** (free GitHub Pages needs a public repository unless you have
   GitHub Pro / the Student pack).
4. Do **not** tick "Add a README", ".gitignore" or "license" — the repository must start empty.
5. Click **Create repository**. Keep this page open; you need its URL
   (it looks like `https://github.com/YOUR-USERNAME/flxai-iiot-reproduction.git`).

---

## Part 5 — Turn on GitHub Pages

1. In your new repository click **Settings** → **Pages** (left menu).
2. Under **Build and deployment → Source**, choose **GitHub Actions**.

If you cannot find this option yet, do it after Part 6 and then re-run the pipeline
(see "If something fails" at the end).

---

## Part 6 — Push the code (first time)

In the VS Code terminal, inside the project folder:

1. Tell Git who you are (only the first time on this computer; use your own name and
   the email of your GitHub account):

   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

2. Create the local repository and the first commit:

   ```
   git init -b main
   git add .
   git commit -m "Initial commit: paper reproduction with CI/CD"
   ```

3. Check the dataset is included — this must print `data/ML-EdgeIIoT-dataset.csv.gz`:

   ```
   git ls-files data
   ```

4. Connect to GitHub and push (replace the URL with **your** repository URL from Part 4):

   ```
   git remote add origin https://github.com/YOUR-USERNAME/flxai-iiot-reproduction.git
   git push -u origin main
   ```

   A browser window may open asking you to sign in to GitHub — sign in and allow access.

---

## Part 7 — Watch the pipeline and open your page

1. On GitHub, open your repository → **Actions** tab. You will see a run called
   **CI/CD** with your commit message.
2. Click it. Three jobs run one after another: **Test → Train → Deploy**.
   Training on the real dataset is the slow part — about 20 minutes when it was
   tested on a 2-core computer; on GitHub it may be a bit faster or slower.
   A yellow dot means running, a green tick means passed, a red cross means failed.
3. When **Deploy** is green, click it — the page link is shown there. It is also
   in **Settings → Pages**. It looks like:
   `https://YOUR-USERNAME.github.io/flxai-iiot-reproduction/`

---

## Part 8 — Make a change and see it on the page

This is how you show the CI/CD working.

1. In VS Code open `src/config.py`. Find `XGB_PARAMS` and change
   `"n_estimators": 215,` to `"n_estimators": 250,`. Save (Ctrl+S).
2. In the terminal:

   ```
   git add .
   git commit -m "Try 250 trees for XGBoost"
   git push
   ```

3. Watch the new run in the **Actions** tab. When Deploy is green, refresh your page
   (Ctrl+F5). The header shows your new commit message, the scores update, and the
   **Settings used for this run** section highlights `n_estimators = 250`
   ("paper: 215").

Every later change is the same three commands: `git add .` → `git commit -m "..."` → `git push`.

---

## Part 9 — Preview the page on your computer (optional)

```
python scripts/run_pipeline.py
cd frontend
npm install
npm run dev
```

Open the `http://localhost:...` address it prints (Ctrl+click). Press `Ctrl+C` in the
terminal to stop it, and `cd ..` to go back to the project folder.

---

## If something fails

- **A job has a red cross:** click it, then the red step, and read the last lines of
  the log. You can paste them to me.
- **Deploy fails with a Pages error:** do Part 5, then open the failed run and click
  **Re-run all jobs**.
- **Train fails with "Dataset missing":** the `.csv.gz` file was not committed. Check
  that `data/ML-EdgeIIoT-dataset.csv.gz` exists, then run
  `git add data/ML-EdgeIIoT-dataset.csv.gz`, commit and push.
- **Train fails with "QUALITY GATE FAILED":** your change made accuracy drop below 0.90.
  That is the pipeline protecting the published page; undo or adjust the change and push again.
- **Run the pipeline again without changing code:** Actions tab → **CI/CD** (left) →
  **Run workflow**.
