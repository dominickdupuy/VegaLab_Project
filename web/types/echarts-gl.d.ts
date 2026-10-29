/**
 * echarts-gl ships no types. Each export is an installer handed to echarts.use();
 * only the ones the 3D views register are declared.
 */
declare module "echarts-gl/charts" {
  type Installer = (registers: unknown) => void;
  export const SurfaceChart: Installer;
  export const Line3DChart: Installer;
  export const Scatter3DChart: Installer;
}

declare module "echarts-gl/components" {
  type Installer = (registers: unknown) => void;
  export const Grid3DComponent: Installer;
}
