/**
 * Shared chart utilities for loss visualization components.
 *
 * This module provides pure functions for chart data computation,
 * used by both LossChart.svelte (small overview) and LossOverTimePage.svelte (full interactive).
 */

import { LARGE_NUMBER_THRESHOLD } from './constants';
import type { HistoryEntry } from './api';

// ============================================================================
// Types
// ============================================================================

export interface ChartPoint {
  x: number;
  y: number;
  step: number;
  value: number;
}

export interface XTick {
  value: number;
  x: number;
}

export interface YTick {
  value: number;
  y: number;
}

export interface ChartPaths {
  total: string;
  policy: string;
  value: string;
  avg100?: string;
}

export interface ChartData {
  xTicks: XTick[];
  yTicks: YTick[];
  paths: ChartPaths;
  points?: {
    total: ChartPoint[];
    policy: ChartPoint[];
    value: ChartPoint[];
  };
}

export interface ChartDimensions {
  width: number;
  height: number;
  padding: { top: number; right: number; bottom: number; left: number };
}

// ============================================================================
// Constants
// ============================================================================

export const CHART_COLORS = {
  total: '#00d9ff', // Cyan - primary brand color
  policy: '#ff6b6b', // Red
  value: '#4ecdc4', // Teal
  avg100: '#ffd700', // Gold
} as const;

// ============================================================================
// Formatting Functions
// ============================================================================

export function formatStep(value: number): string {
  const million = LARGE_NUMBER_THRESHOLD * LARGE_NUMBER_THRESHOLD;
  if (value >= million) return `${(value / million).toFixed(1)}M`;
  if (value >= LARGE_NUMBER_THRESHOLD)
    return `${(value / LARGE_NUMBER_THRESHOLD).toFixed(1)}k`;
  return Math.round(value).toString();
}

export function formatLoss(value: number): string {
  if (value < 0.001) return value.toExponential(2);
  if (value < 0.01) return value.toFixed(4);
  if (value < 1) return value.toFixed(3);
  return value.toFixed(2);
}

// ============================================================================
// Scale & Tick Functions
// ============================================================================

export function generateTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

export function createScales(
  data: HistoryEntry[],
  chartWidth: number,
  chartHeight: number,
  includeAvg100Data?: { step: number; avg: number }[]
): {
  xScale: (step: number) => number;
  yScale: (loss: number) => number;
  minStep: number;
  maxStep: number;
  yMin: number;
  yMax: number;
} {
  const steps = data.map((d) => d.step);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);

  const allLosses = data.flatMap((d) => [d.metrics['loss/total'], d.metrics['loss/policy'], d.metrics['loss/value']]);
  if (includeAvg100Data && includeAvg100Data.length > 0) {
    allLosses.push(...includeAvg100Data.map((d) => d.avg));
  }

  const maxLoss = Math.max(...allLosses);
  const minLoss = Math.min(...allLosses);
  const yMax = maxLoss === 0 ? 1 : maxLoss * 1.1;
  const yMin = Math.max(0, minLoss * 0.9);
  const yRange = yMax - yMin || 1;

  const xScale = (step: number) => {
    if (maxStep === minStep) return chartWidth / 2;
    return ((step - minStep) / (maxStep - minStep)) * chartWidth;
  };

  const yScale = (loss: number) => {
    return chartHeight - ((loss - yMin) / yRange) * chartHeight;
  };

  return { xScale, yScale, minStep, maxStep, yMin, yMax };
}

// ============================================================================
// Path Generation
// ============================================================================

export function makePath(
  data: HistoryEntry[],
  key: 'loss/total' | 'loss/policy' | 'loss/value',
  xScale: (step: number) => number,
  yScale: (loss: number) => number
): string {
  const sorted = [...data].sort((a, b) => a.step - b.step);
  return sorted
    .map((d, i) => {
      const x = xScale(d.step);
      const y = yScale(d.metrics[key]);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');
}

export function makeAvg100Path(
  avg100Data: { step: number; avg: number }[],
  xScale: (step: number) => number,
  yScale: (loss: number) => number
): string {
  if (avg100Data.length === 0) return '';
  return avg100Data
    .map((d, i) => {
      const x = xScale(d.step);
      const y = yScale(d.avg);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');
}

// ============================================================================
// Rolling Average
// ============================================================================

export function computeRollingAverage(
  data: HistoryEntry[],
  window: number = 100
): { step: number; avg: number }[] {
  if (data.length === 0) return [];
  const sorted = [...data].sort((a, b) => a.step - b.step);
  const result: { step: number; avg: number }[] = [];

  for (let i = 0; i < sorted.length; i++) {
    const start = Math.max(0, i - window + 1);
    const windowData = sorted.slice(start, i + 1);
    const avg = windowData.reduce((sum, d) => sum + d.metrics['loss/total'], 0) / windowData.length;
    result.push({ step: sorted[i].step, avg });
  }
  return result;
}

// ============================================================================
// Main Chart Data Builder
// ============================================================================

export interface BuildChartDataOptions {
  data: HistoryEntry[];
  chartWidth: number;
  chartHeight: number;
  xTickCount?: number;
  yTickCount?: number;
  includePoints?: boolean;
  includeAvg100?: boolean;
}

export function buildChartData(options: BuildChartDataOptions): ChartData {
  const {
    data,
    chartWidth,
    chartHeight,
    xTickCount = 5,
    yTickCount = 5,
    includePoints = false,
    includeAvg100 = false,
  } = options;

  if (data.length === 0) {
    return {
      xTicks: [],
      yTicks: [],
      paths: { total: '', policy: '', value: '' },
      points: includePoints ? { total: [], policy: [], value: [] } : undefined,
    };
  }

  const sorted = [...data].sort((a, b) => a.step - b.step);
  const avg100Data = includeAvg100 ? computeRollingAverage(sorted) : [];

  const { xScale, yScale, minStep, maxStep, yMin, yMax } = createScales(
    sorted,
    chartWidth,
    chartHeight,
    includeAvg100 ? avg100Data : undefined
  );

  const paths: ChartPaths = {
    total: makePath(sorted, 'loss/total', xScale, yScale),
    policy: makePath(sorted, 'loss/policy', xScale, yScale),
    value: makePath(sorted, 'loss/value', xScale, yScale),
  };

  if (includeAvg100) {
    paths.avg100 = makeAvg100Path(avg100Data, xScale, yScale);
  }

  const xTicks = generateTicks(minStep, maxStep, xTickCount).map((v) => ({
    value: v,
    x: xScale(v),
  }));

  const yTicks = generateTicks(yMin, yMax, yTickCount).map((v) => ({
    value: v,
    y: yScale(v),
  }));

  let points: ChartData['points'];
  if (includePoints) {
    const makePoints = (key: 'loss/total' | 'loss/policy' | 'loss/value') =>
      sorted.map((d) => ({
        x: xScale(d.step),
        y: yScale(d.metrics[key]),
        step: d.step,
        value: d.metrics[key],
      }));

    points = {
      total: makePoints('loss/total'),
      policy: makePoints('loss/policy'),
      value: makePoints('loss/value'),
    };
  }

  return { xTicks, yTicks, paths, points };
}
