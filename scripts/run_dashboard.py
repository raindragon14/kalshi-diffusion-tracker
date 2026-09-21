#!/usr/bin/env python3
"""Generate static dashboard"""
import logging

from src.dashboard.generate import generate_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Generating dashboard...")
    generate_dashboard()
    logger.info("Dashboard generation complete")


if __name__ == "__main__":
    main()
