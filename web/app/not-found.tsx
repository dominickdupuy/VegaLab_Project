import Link from "next/link";
import { Empty } from "@/components/States";

export default function NotFound() {
  return (
    <Empty title="Nothing here">
      <p>
        <Link href="/">Overview</Link>
      </p>
    </Empty>
  );
}
