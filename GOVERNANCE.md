# ModelKit Governance

ModelKit is maintained by Asara and provided as open source software 
under the respective license (see `LICENSE.md`). 

This document describes how responsibility and decision authority are
exercised. The repository's configured maintainer team is the authoritative
record of current maintainers.

## Roles

Contributors propose changes, report defects, review work, and participate in
technical discussion. Any participant may become a contributor by engaging
constructively with the project.

Maintainers are contributors with repository write access. They are
responsible for technical review, repository health, support triage, and
ensuring changes follow the product plan and quality requirements.

The product owner controls product scope, milestone sign-off, release
declarations, and the public-contract commitments attached to a release.

## Decisions and Changes

Routine changes are decided through review (primarily through Pull Requests 
and Issues) in the repository. A change should be small enough to review 
coherently and should include the evidence needed to show that its stated 
acceptance criteria pass. Approval indicates agreement that the change is 
correct, appropriately scoped, and maintainable. This policy applies to 
human-written and machine-written/AI-written changes equally.

Changes to public APIs, artifact formats, numerical semantics, portability,
dependencies, licensing, security posture, governance, or release scope
require explicit maintainer review. Material changes to product scope or a
milestone require product-owner approval. Important decisions should be
recorded in the repository so later contributors can recover their rationale.

The product owner makes the final scope or release decision, in the event 
the maintainers do not reach consensus. A maintainer makes the final 
technical-health decision when a proposal would violate a supported contract 
or quality gate.

## Reviews and Releases

Changes are merged only after their relevant build, test, documentation, and
compatibility checks pass. Maintainer-authored changes should receive review
from another maintainer when one is available. Contributors must disclose
known limitations, deferred verification, and conflicts of interest relevant
to a decision.

Maintainers prepare releases, but only the product owner may declare a
milestone complete or authorize a release. Release artifacts must match the
reviewed source and the support policy in effect for that release.

## Participation

Project communication must be respectful, technically constructive, and
welcoming. Maintainers may moderate participation to protect contributors 
and the project.

Governance changes use the same review process as other material changes.
The product owner approves changes to decision authority.
