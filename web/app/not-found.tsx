import Link from "next/link";
import { Empty } from "@/components/States";

export default function NotFound() {
  return (
    <Empty title="Nothing here">
      <p>
        This page does not exist. <Link href="/">Back to the overview</Link>.
      </p>
    </Empty>
  );
}
