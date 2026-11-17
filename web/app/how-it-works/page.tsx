"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

/**
 * How it works is retired: Home now tells the basics, the problem and the solution. The
 * site is a static export, so there is no server redirect: this page replaces itself with
 * / and leaves a link in case script is off, so old links still land somewhere.
 */
export default function HowItWorks() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/");
  }, [router]);
  return (
    <p>
      How it works is now part of <Link href="/">Home</Link>.
    </p>
  );
}
