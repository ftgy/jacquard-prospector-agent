# Connecting Gmail (one-time setup)

The Outreach tab sends cold emails through **your own Gmail account** and tracks
which ones got replies. To do that, the app needs two files in a **`secrets/`**
directory at the project root, both git-ignored:

| File | What it is | How you get it |
| --- | --- | --- |
| `secrets/credentials.json` | The OAuth *client* — identifies this app to Google. | Downloaded from Google Cloud (steps below). |
| `secrets/token.json` | Your *authorization* — lets the app act as you. | Minted by `scripts/gmail_auth.py`. |

You only do the Google Cloud part once. It takes about ten minutes.

## 1. Create a Google Cloud project

1. Go to <https://console.cloud.google.com/> and create a new project (top bar →
   project dropdown → **New project**). Name it anything, e.g. `prospector`.
2. With that project selected, open **APIs & Services → Library**, search for
   **Gmail API**, and click **Enable**.

## 2. Configure the OAuth consent screen

1. **APIs & Services → OAuth consent screen.**
2. User type **External**, then **Create**.
3. Fill in the required fields (app name, your email as the support + developer
   contact). You can leave everything optional blank. **Save and continue.**
4. **Scopes** — skip (we request scopes from code). **Save and continue.**
5. **Test users → Add users** — add the Gmail address you'll send from. This is
   important: while the app is in "testing", only listed test users can authorize
   it. **Save and continue.**

You do **not** need to publish the app or go through Google verification — testing
mode is fine for personal use. (Refresh tokens in testing mode can expire after a
few weeks of disuse; if sending ever stops working, just re-run step 4 below.)

## 3. Create the OAuth client

1. **APIs & Services → Credentials → Create credentials → OAuth client ID.**
2. Application type **Desktop app**. Name it anything. **Create.**
3. In the dialog, **Download JSON**. Save it as `secrets/credentials.json` in the
   project (create the `secrets/` folder if it isn't there — it's git-ignored).

## 4. Authorize your account

From the project root, with the virtualenv active:

```bash
pip install -r requirements.txt          # first time, for the Google libraries
python scripts/gmail_auth.py
```

A browser window opens asking you to sign in and grant access. Approve it. (If you
see an "unverified app" warning, that's expected for a testing-mode app — continue
past it.) On success the script writes `secrets/token.json` and prints the
connected account.

Check it any time without changing anything:

```bash
python scripts/gmail_auth.py --status
```

## 5. Send

Start the server (`python run_server.py`), open the **Outreach** tab — it should
show **Gmail connected** with your address. Now the **Send** button on any
prospect's drafted email sends it through your account, and the tab tracks sends
and replies. Use **Check replies** to poll your sent threads for responses.

## Scopes we request

Least privilege — just what the two features need:

- `gmail.send` — send mail as you.
- `gmail.readonly` — read a thread to see whether someone replied.

## If something breaks

- **"Gmail isn't connected"** — no `token.json`, or it expired. Run
  `python scripts/gmail_auth.py` again.
- **"access_denied" during authorization** — the account isn't in the project's
  test-user list (step 2.5), or it isn't the account you signed in with.
- **Changing scopes** — editing `SCOPES` in `prospector/gmailer.py` invalidates
  the old token; re-run the auth script.

Never commit `credentials.json` or `token.json` — both are in `.gitignore`.
