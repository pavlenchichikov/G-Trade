-- Atratus mobile app backend: run once in the Supabase SQL editor.
-- Adds device push tokens plus four read-only snapshot tables for the
-- Flutter client. Gated tables reuse the same allow-list gate as `signals`.

create table if not exists device_tokens (
  token      text primary key,
  user_id    uuid not null references auth.users(id) on delete cascade,
  updated_at timestamptz default now()
);
alter table device_tokens enable row level security;
drop policy if exists dt_own on device_tokens;
create policy dt_own on device_tokens
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create table if not exists bars (
  asset text not null,
  date  date not null,
  open double precision, high double precision,
  low  double precision, close double precision,
  primary key (asset, date)
);

create table if not exists signal_history (
  asset text not null,
  date  date not null,
  signal text,
  prob double precision,
  actual_next_ret double precision,
  correct integer,
  primary key (asset, date)
);

create table if not exists guru (
  asset text primary key,
  verdict text,
  council_pct double precision,
  lynch integer, buffett integer, graham integer, munger integer,
  source text,
  date date,
  correct_5d integer
);

create table if not exists guru_stats (
  id integer primary key,
  accuracy double precision,
  n integer,
  horizon text
);

alter table bars enable row level security;
alter table signal_history enable row level security;
alter table guru enable row level security;
alter table guru_stats enable row level security;

drop policy if exists bars_read on bars;
create policy bars_read on bars for select using (is_allowed());
drop policy if exists signal_history_read on signal_history;
create policy signal_history_read on signal_history for select using (is_allowed());
drop policy if exists guru_read on guru;
create policy guru_read on guru for select using (is_allowed());
drop policy if exists guru_stats_read on guru_stats;
create policy guru_stats_read on guru_stats for select using (is_allowed());

create or replace function allowed_device_tokens()
returns table (token text) language sql security definer as $$
  select dt.token
  from device_tokens dt
  join auth.users u on u.id = dt.user_id
  join access_list a on lower(a.email) = lower(u.email);
$$;
revoke execute on function allowed_device_tokens() from anon, authenticated;
