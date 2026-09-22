# Sending cold email without landing on a spam list

The rules we follow so outreach from `hello@feina.dev` keeps reaching inboxes. The
short version: **send few, send slowly, make leaving easy, and stop the moment
anyone asks.**

## The two things that get you blocked

Spam filters (Gmail, Outlook, corporate filters) and blocklists mostly react to
two signals:

1. **Spam complaints.** When a recipient clicks "Report spam". Google wants the
   complaint rate **below 0.1%** and starts penalising at **0.3%**. At our volume,
   one or two reports in a week is already a bad sign.
2. **Bounces.** Mail sent to addresses that don't exist. Keep hard bounces
   **below 2%**. A burst of bounces looks like a bought or guessed list.

Everything below exists to keep those two numbers down.

An "unsubscribe" reply is actually a good outcome: the person told *you* instead
of clicking "Report spam". Honour it right away and it costs you nothing.

## When someone asks to unsubscribe

- **Stop every email to them, today.** Once "Check replies" sees their reply, the
  prospect is marked replied and gets no more follow-ups. Make sure that has
  happened.
- **Never email that company again**, including from a future populate run that
  finds it a second time. For now, keep a note of it (or **Remove** the prospect
  and check before you send to anyone new). A real do-not-contact list in the app
  is still a TODO.
- If you reply at all, keep it to one line: *"Hecho, no te vuelvo a escribir.
  Disculpa la molestia."* Don't argue and don't pitch.

## How many emails a day

This is one mailbox on a young domain, so reputation builds slowly and is easy
to lose.

The dashboard doesn't send anything itself. **Queue all** hands drafts to the
prospector-scheduler service, and that service applies the limits and times below
when it picks each email's slot. The settings are `SCHEDULER_DAILY_CAP`,
`SCHEDULER_SEND_WINDOWS`, `SCHEDULER_BLACKOUT_DATES` and
`SCHEDULER_GAP_MIN/MAX_MINUTES` in the scheduler's `.env`. To raise the daily
limit, change `SCHEDULER_DAILY_CAP` there.

| Period | New first emails per day | Notes |
| --- | --- | --- |
| Weeks 1–2 | **10–15** | Watch every reply and bounce by hand. |
| Weeks 3–4 | **20–30** | Only if bounces < 2% and nobody reported spam. |
| After that | **30–50 max** | This is the ceiling for one mailbox. For more volume, add mailboxes, don't push this one harder. |

- Follow-ups count toward the day's total. With `FOLLOWUP_DAYS=4,7`, 20 new emails a
  day grows to about 40–50 sends a day once follow-ups start going out.
- **Space sends out**, a few minutes apart at irregular intervals. The scheduler
  does this automatically, 4 to 9 minutes apart.
- **Keep it steady.** 15 a day every working day is better than 0, 0, 60.
- Gmail's hard limits (500/day personal, 2,000/day Workspace) are not a target.
  Getting close to them is how accounts get suspended.

## Which days and times (Spain)

Send so the email arrives just before the person sits down to read their inbox.

- **Best days:** Tuesday, Wednesday, Thursday.
- **Best times (Madrid time):** **08:30–10:30**, then **16:00–17:30** when people are
  back from lunch.
- **Avoid:** Monday morning (weekend backlog), Friday afternoon, weekends, and
  14:00–16:00.
- **Avoid these periods completely:** August, 23 Dec – 7 Jan, Semana Santa, and
  local holidays in the prospect's city (Fallas in Valencia, Sant Joan in
  Catalonia, San Isidro in Madrid…). Mail sent then gets buried or read in a bad
  mood.

## Content rules (the playbook already covers most)

- Plain text, no attachments, no images, no tracking pixel, no links in the body.
  The playbook already works this way.
- Each email is personalised from the research. Mass-identical bodies get
  fingerprinted.
- One clear question at the end, not a list of offers.
- **Add a soft opt-out line** at the bottom, in the same tone as the rest:
  *"Si no te encaja, dímelo y no te escribo más."* It gives people an easy way out
  that isn't the spam button, and it's what the law asks for (see below).
- At most **two follow-ups**, in the same thread, and none after any reply.

## List quality

- Only send to addresses the research found published on the company's own site or
  legal notice. **Never guess** addresses like `nombre@empresa.es`.
- A generic contact address (`info@`, `hola@`, `contacto@`) is fine, and in Spain
  it's often the only one there is.
- If a send bounces, stop sending to that company. Don't retry with a variant.

## Domain authentication: to check

Filters trust mail more when the `From` domain matches the domain the mail is
signed with. A DNS check of `feina.dev` (2026-09-22) found:

- **MX:** Cloudflare Email Routing, so replies to `hello@feina.dev` arrive. ✅
- **SPF:** `v=spf1 include:_spf.mx.cloudflare.net ~all`. This only authorises
  Cloudflare, and Cloudflare doesn't send mail. Google isn't in it. ⚠️
- **DMARC:** no `_dmarc.feina.dev` record. ⚠️
- **DKIM:** no `google._domainkey.feina.dev` key. The mail goes out through
  Gmail, so it's signed as `gmail.com`, not `feina.dev`. ⚠️

What this likely means: recipients see "via gmail.com", and neither SPF nor DKIM
lines up with `feina.dev`. That doesn't block delivery by itself, but it's a
negative signal. To confirm, send an email to a Gmail address you own, open **⋮ →
Show original**, and read the SPF / DKIM / DMARC lines. You can also send one to
<https://www.mail-tester.com> for a score.

To fix it, in order of effort:

1. Add a DMARC record in monitor-only mode:
   `_dmarc.feina.dev TXT "v=DMARC1; p=none; rua=mailto:hello@feina.dev"`.
2. Send `hello@feina.dev` through a provider that signs with DKIM for
   `feina.dev`: Google Workspace for the domain, or an SMTP provider configured as
   the "Send mail as" server. Add that provider's SPF include and DKIM key in
   Cloudflare DNS.
3. For much higher volume later, send cold email from a **separate domain**
   (e.g. `feina.io`) so a bad week can't hurt the main one.

## The legal side (Spain)

Not legal advice, but you should know this. Spain's LSSI (art. 21) forbids
unsolicited commercial email, **and that applies to companies too**, not only to
consumers. The AEPD has fined B2B cold email. What reduces the risk:

- Write to a specific business, about that business, from a real person.
  Personalised, low-volume email is far from mass mailing.
- Say clearly who you are (the signature does this).
- **Always give a way to opt out, and honour it immediately and permanently.**
- Keep a record of every opt-out.

## Signals to watch weekly

| Signal | Healthy | Stop and review |
| --- | --- | --- |
| Bounces | < 2% | ≥ 3% |
| Unsubscribe / annoyed replies | ≤ 1 in 50 | several in a week |
| "Report spam" (Gmail may warn you) | 0 | any |
| Reply rate | rising or stable | sudden drop to ~0 (you may be in spam) |

If replies stop completely, send one test email to your own Gmail and Outlook
addresses and check whether it lands in spam before you keep sending.
