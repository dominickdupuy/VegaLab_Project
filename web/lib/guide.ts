/**
 * The hook a workflow step points at. A page marks the control a step asks for with
 * `data-guide="<name>"`, most simply by spreading this helper:
 *
 *   <button {...guide("launch")} onClick={launch}>Launch run</button>
 *
 * The workflow strip finds the control by that name first (and by its text or label
 * when no page has the hook yet), scrolls it into view and rings it: three pulses, then
 * still, until the step is done. The names in use are listed in lib/tours.ts.
 */
export const GUIDE_ATTR = "data-guide";
/** Set by the strip on the control it is pointing at; styled in app/globals.css. */
export const GUIDE_ON_ATTR = "data-guide-on";

export function guide(name: string): { "data-guide": string } {
  return { "data-guide": name };
}
