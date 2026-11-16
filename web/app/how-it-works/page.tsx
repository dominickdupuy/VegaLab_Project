"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

/**
 * How it works moved into the tutorial. The site is a static export, so there is no server
 * redirect: this page replaces itself with /tutorial/ and leaves a link in case script is off.
 */
export default function HowItWorks() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/tutorial/");
  }, [router]);
  return (
    <p>
      How it works is now part of the <Link href="/tutorial/">tutorial</Link>.
    </p>
  );
}
