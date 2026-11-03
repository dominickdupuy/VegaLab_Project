import type { Metadata } from "next";
import Link from "next/link";
import { Card, PageHead } from "@/components/Bits";
import ui from "@/components/ui.module.css";

export const metadata: Metadata = { title: "How it works" };

// ---------------------------------------------------------------- pipeline

const PIPELINE_STEPS: { label: string; detail: string }[] = [
  { label: "Market Config", detail: "alpha_bps, vol_of_vol, skew_slope, sigma" },
  { label: "Synthetic Chains", detail: "Zero-cost Black-Scholes chains, per-day" },
  { label: "RL Environment", detail: "Obs: spot, IV surface, Greeks, P&L" },
  { label: "Train / Run Arm", detail: "PPO, A2C, DQN or scripted baseline" },
  { label: "Equity Curves", detail: "N episodes × T days P&L paths" },
  { label: "Compare & Iterate", detail: "Adjust knobs, re-run, compare in dashboard" },
];

function PipelineDiagram() {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "0.5rem",
        alignItems: "center",
        margin: "0.75rem 0 0.25rem",
      }}
    >
      {PIPELINE_STEPS.map((s, i) => (
        <div key={s.label} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div
            style={{
              padding: "0.75rem 1rem",
              background: "var(--surface-2)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-sm)",
              minWidth: 140,
            }}
          >
            <div style={{ fontWeight: 700, fontSize: "0.875rem" }}>{s.label}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: 3 }}>{s.detail}</div>
          </div>
          {i < PIPELINE_STEPS.length - 1 && (
            <span style={{ fontSize: "1.25rem", color: "var(--muted)", userSelect: "none" }} aria-hidden="true">
              →
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- knobs table

const KNOBS: { name: string; type: string; default: string; description: string }[] = [
  {
    name: "alpha_bps",
    type: "float ≥ 0",
    default: "0",
    description:
      "Per-day drift in bps. 0 = zero-drift (only theta-selling has edge). 25 ≈ SPX long-bias. Use to inject a known signal and verify models learn it.",
  },
  {
    name: "vol_of_vol",
    type: "float ≥ 0",
    default: "0",
    description:
      "Stochastic vol intensity. 0 = constant vol (GBM). 0.3 = realised-vol-like variation. Higher values stress-test IV surface modelling.",
  },
  {
    name: "skew_slope",
    type: "float",
    default: "0",
    description:
      "Log-moneyness IV tilt. Positive = equity put skew (low-strike IV higher, puts cost more). Tests whether models hedge skew exposure correctly.",
  },
  {
    name: "sigma",
    type: "float > 0",
    default: "0.20",
    description: "Base annualised volatility (20% ≈ SPX long-run average).",
  },
  {
    name: "n_days",
    type: "int > 0",
    default: "63",
    description: "Episode length in trading days. Shorter = faster experiments.",
  },
];

function KnobsTable() {
  return (
    <div className={ui.tableWrap}>
      <table className={ui.table}>
        <thead>
          <tr>
            <th>Param</th>
            <th>Type</th>
            <th>Default</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          {KNOBS.map((k) => (
            <tr key={k.name}>
              <td className="mono">{k.name}</td>
              <td className={ui.dim}>{k.type}</td>
              <td className="mono">{k.default}</td>
              <td>{k.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- arms table

const ARMS: { name: string; kind: "Learned" | "Scripted"; description: string }[] = [
  {
    name: "ppo",
    kind: "Learned",
    description:
      "Proximal Policy Optimisation (Stable-Baselines3). Trains from scratch each run. Use to test whether RL can discover the available edge.",
  },
  {
    name: "a2c",
    kind: "Learned",
    description:
      "Advantage Actor-Critic. Faster per-step than PPO, often noisier. Good for quick hypothesis tests.",
  },
  {
    name: "dqn",
    kind: "Learned",
    description:
      "Deep Q-Network. Discrete actions only. Use to compare value-based vs. policy-gradient approaches.",
  },
  {
    name: "delta_seller",
    kind: "Scripted",
    description:
      "Sells a fixed-delta strangle each day. Structural theta-capture baseline with no learning.",
  },
  {
    name: "flat",
    kind: "Scripted",
    description:
      "Holds a flat position throughout. Simple baseline to confirm the market is not trivially profitable.",
  },
  {
    name: "always_on",
    kind: "Scripted",
    description: "Keeps the condor position open at all times. Tests pure theta exposure.",
  },
  {
    name: "random",
    kind: "Scripted",
    description: "Random action each step. Lower bound on performance.",
  },
];

function ArmsTable() {
  return (
    <div className={ui.tableWrap}>
      <table className={ui.table}>
        <thead>
          <tr>
            <th>Arm</th>
            <th>Kind</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {ARMS.map((a) => (
            <tr key={a.name}>
              <td className="mono">{a.name}</td>
              <td>
                <span
                  style={{
                    color:
                      a.kind === "Learned" ? "var(--accent-text)" : "var(--muted)",
                    fontWeight: 600,
                    fontSize: "0.8125rem",
                  }}
                >
                  {a.kind}
                </span>
              </td>
              <td>{a.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- page

export default function HowItWorksPage() {
  return (
    <>
      <PageHead
        eyebrow="Explainer"
        title="How VegaLab works"
        lede="VegaLab is a research instrument for testing RL policies on options markets. It generates synthetic SPX-like option chains, trains or runs arms against them, and lets you compare equity curves across strategies and market regimes."
      />

      <Card title="Research pipeline" sub="Data flows left to right through these six stages every run.">
        <PipelineDiagram />
      </Card>

      <Card title="Market knobs" sub="Configure these in the Launch panel to shape the synthetic market.">
        <KnobsTable />
      </Card>

      <Card title="Arms" sub="Each run trains or runs one arm against the synthetic market for N episodes.">
        <ArmsTable />
      </Card>

      <Card title="Reading the equity curves">
        <p style={{ color: "var(--text-2)", lineHeight: 1.6 }}>
          The equity chart shows cumulative P&amp;L across trading days for each episode seed. A flat line
          means no edge was found. An upward slope means the arm is capturing the available alpha. The
          thick line is the mean across seeds; thin lines are individual seeds.
        </p>
        <p style={{ color: "var(--text-2)", lineHeight: 1.6, marginTop: "0.75rem" }}>
          Compare arms at the same{" "}
          <code>alpha_bps</code> to see which learns fastest. Compare at{" "}
          <code>alpha_bps&nbsp;=&nbsp;0</code> to measure pure theta-capture edge.
        </p>
      </Card>

      <div
        className={ui.callout}
        style={{ marginTop: 0 }}
        role="note"
      >
        <strong>Try it:</strong> Go to the{" "}
        <Link href="/launch/">Launch panel</Link>, set{" "}
        <code>alpha_bps = 25</code> and run PPO vs. delta_seller to see which arm discovers the edge first.
      </div>
    </>
  );
}
