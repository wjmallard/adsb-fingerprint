"""Top-level command: print an overview of the adsb-fingerprint toolbox."""

OVERVIEW = """\
adsb-fingerprint: RF fingerprinting of ADS-B transponders from RTL-SDR captures

  adsb-info            print RTL-SDR device details and the capture target
  adsb-initdb          apply sql/ to the configured Postgres database
  adsb-collect         stream the SDR, persist only detected Mode S snippets
  adsb-capture         stream raw IQ into timestamped .cf32 files
  adsb-index           (re)detect messages in captures and fill the index
  adsb-gps-log         log GPS fixes from the USB puck to daily track files
  adsb-geotag          back-fill receiver positions from GPS track logs
  adsb-ingest-faa      load the FAA Releasable Aircraft registry
  adsb-ingest-opensky  load the OpenSky aircraft database CSV
  adsb-stats           summarize the database
  adsb-dataset         summarize the fingerprinting dataset built from the index
  adsb-train           train the ADCC model with whole sessions held out
  adsb-eval            cross-session evaluation, ablation comparison, baselines
  adsb-verify          open-set evaluation: verification EER, stranger AUROC
  adsb-embed           write a run's self-contained embedding viewer
  adsb-variance        decompose feature variance (message / window / session)
  adsb-predict         continuously score messages against enrolled aircraft
  adsb-web             live map, roster, detail panel, and embedding viewers
  adsb                 show this overview

Most commands take --help for options. Setup (database, config.yaml,
registry data) is covered in README.md.
"""


def main():
    print(OVERVIEW, end="")


if __name__ == "__main__":
    main()
