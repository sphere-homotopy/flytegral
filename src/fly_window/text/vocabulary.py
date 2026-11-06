from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


SPECIAL_TOKENS = (
    "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>", "<SEP>", "<NL>", "<URL>",
    "<USER>", "<NUMBER>", "<INTEGER>", "<REAL>", "<MATH>", "<CODE>", "<QUOTE>", "<EMPH>",
    "<OPEN>", "<CLOSE>", "<LIST>", "<EQ>", "<NEQ>", "<LT>", "<GT>", "<LE>",
    "<GE>", "<ARROW>", "<IFF>", "<AND>", "<OR>", "<NOT>", "<TRUE>", "<FALSE>",
)

SYMBOL_TOKENS = (
    ".", ",", "?", "!", ":", ";", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/",
    "=", "<", ">", "<=", ">=", "!=", "~", "^", "_", "|", "&", "%", "#", "@", "$", "'",
    "\"", "`", "...", "->", "<-", "<->", "=>", "<=>", "∈", "∉", "⊂", "⊆", "⊃", "⊇", "∪", "∩",
    "∅", "∞", "∑", "∏", "∫", "∂", "∇", "√", "±", "≈", "≡", "⊕", "⊗", "∀", "∃", "∎",
)

_GENERAL_SOURCE = """
a an the this that these those i you we they he she it me us them my your our their
is are am was were be been being have has had do does did can could may might must
will would should shall not no yes if then else when while because so but and or nor
for to from of in on at by with without about into onto over under between among
through across around before after during since until as than like unlike very more
most less least much many few some any all every each either neither both another
same other own such only even also just still already yet again once always never
often sometimes usually maybe perhaps probably certainly really actually basically
apparently literally almost exactly roughly approximately simply clearly obviously
truly deeply mostly partly enough too quite rather pretty kind sort somehow
what which who whom whose where why how whether here there now today tomorrow yesterday
time day night morning evening moment year month week hour minute second
thing things idea ideas question questions answer answers problem problems result results
reason reasons example examples case cases way ways part parts point points side sides
world worlds place places object objects process processes system systems structure structures
one two three four five six seven eight nine ten first second third last next previous
small large big tiny huge high low long short old new good bad better worse best worst
simple hard easy difficult possible impossible true false right wrong real fake
different similar equal common rare general special local global finite infinite
open closed inside outside left right above below near far together apart
make makes made take takes took give gives gave get gets got use uses used
find finds found show shows shown prove proves proved think thinks thought know knows known
say says said ask asks asked write writes wrote read reads see sees saw
mean means meant need needs needed want wants wanted try tries tried work works worked
change changes changed keep keeps kept start starts started stop stops stopped
move moves moved follow follows followed choose chooses chosen sample samples sampled
learn learns learned train trains trained predict predicts predicted generate generates generated
exist exists existed contain contains contained map maps mapped preserve preserves preserved
converge converges converged diverge diverges diverged imply implies implied
random deterministic stable unstable exact approximate canonical natural generic
human humans machine machines model models data code program programs computer computers
language languages word words token tokens text texts sentence sentences tweet tweets post posts
number numbers set sets function functions relation relations graph graphs space spaces
value values state states input inputs output outputs weight weights matrix matrices vector vectors
"""

_MATH_SOURCE = """
theorem proof lemma corollary proposition conjecture axiom definition counterexample contradiction
hypothesis conclusion assumption condition statement claim argument derivation identity inequality equation formula
variable constant parameter coefficient term expression operator operation inverse composition commutator determinant trace rank
scalar tensor covector basis dimension coordinate coordinates kernel image cokernel nullity eigenvalue eigenvalues eigenvector eigenvectors
spectrum spectral singular regular diagonal diagonalizable orthogonal orthonormal unitary hermitian symmetric antisymmetric positive negative
semidefinite bilinear multilinear quadratic linear nonlinear affine convex concave polynomial monomial binomial rational irrational algebraic transcendental
root roots zero zeros pole poles residue derivative derivatives differential differentials gradient divergence curl jacobian hessian
integral integrals integrand measure measurable measurability lebesgue riemann stieltjes borel sigma epsilon delta limit limits
continuity continuous discontinuity discontinuous uniform pointwise differentiable smooth analytic holomorphic meromorphic harmonic subharmonic
sequence sequences series summation product convergence divergent cauchy complete completion compact compactness bounded unbounded
supremum infimum maximum minimum extrema monotone increasing decreasing oscillation variation absolute norm norms seminorm metric metrics
distance diameter neighborhood neighborhoods ball balls sphere spheres interval intervals domain codomain range preimage fiber fibers
topology topological homeomorphism homeomorphic homotopy homotopic isotopy manifold manifolds variety varieties scheme schemes
open-set closed-set closure interior boundary frontier connected connectedness path path-connected simply-connected contractible
compactification quotient subspace product-space wedge smash suspension loop loops fundamental group groups subgroup subgroups
normal quotient-group cyclic abelian nonabelian finite-group lie-group topological-group representation representations character characters
ring rings field fields module modules ideal ideals prime maximal localization local-ring polynomial-ring algebra algebras
homomorphism isomorphism automorphism endomorphism monomorphism epimorphism morphism morphisms object category categories functor functors
natural-transformation adjunction adjoint monad comonad limit-colimit colimit universal universal-property yoneda topos sheaf sheaves
presheaf presheaves stalk stalks section sections bundle bundles vector-bundle tangent cotangent fiber-bundle principal-bundle
homology cohomology chain cochain complex complexes exact exactness sequence-exact boundary-map coboundary cycle cycles cocycle cocycles
betti euler characteristic degree index orientation oriented differential-form forms de-rham stokes gauss green frobenius
curvature gaussian sectional ricci scalar-curvature connection connections christoffel geodesic geodesics metric-tensor riemannian pseudo-riemannian
symplectic poisson hamiltonian lagrangian contact kahler complex-manifold surface surfaces curve curves hypersurface embedding immersion
cover covering cover-space deck transformation transformations knot knots link links braid braids genus torus tori simplex simplices
simplicial complex-simplicial cell cellular cw-complex triangulation subdivision chaikin refinement mesh polyhedron polyhedra tetrahedron
geometry geometric euclidean hyperbolic spherical projective affine-geometry incidence angle angles length area volume
probability probabilistic random-variable expectation expected variance covariance correlation independence independent conditional conditioning
distribution distributions density mass likelihood posterior prior bayes bayesian frequentist estimator estimation unbiased bias consistency sufficient
statistic statistics sample sampling population mean median mode quantile percentile moment moments cumulant gaussian normal bernoulli
binomial-distribution poisson-distribution exponential-distribution gamma beta-distribution uniform-distribution markov martingale stopping-time filtration
stochastic process-process brownian diffusion random-walk chain chains ergodic ergodicity entropy information mutual-information kl-divergence
law large-numbers central-limit concentration hoeffding chernoff chebyshev almost-surely almost-sure probability-one
combinatorics combinatorial permutation permutations combination combinations partition partitions graph-theory vertex vertices edge edges
tree trees forest matching coloring chromatic clique independent-set planar planarity eulerian hamiltonian-cycle path-count recurrence generating-function
number-theory integer integers divisibility divisor divisors prime primes composite gcd lcm congruence modular modulus residue-class
diophantine pell fermat euler totient mobius inversion quadratic-residue reciprocity continued-fraction rational-approximation transcendence
analysis functional-analysis operator-theory banach hilbert sobolev distribution-theory fourier transform laplace z-transform wavelet
pde ode differential-equation differential-equations navier-stokes heat-equation wave-equation poisson-equation boundary-value initial-value
dynamical dynamics flow flows orbit orbits fixed-point periodic chaos chaotic attractor bifurcation stability lyapunov
optimization optimize optimum gradient-descent newton simplex-method linear-programming convexity duality primal dual lagrange-multiplier
logic logical proposition predicate quantifier quantifiers forall exists negation conjunction disjunction implication equivalence
set-theory cardinal cardinality ordinal ordinals countable uncountable continuum choice zorn transfinite forcing independence-result
computability computable undecidable decidable algorithm algorithms complexity polynomial-time exponential-time np p np-complete reduction reductions
automaton automata turing machine-learning recursion recursive lambda calculus type types type-theory proof-theory model-theory
physics physical mechanics classical quantum relativity spacetime energy momentum force forces mass velocity acceleration action
symmetry symmetries invariant invariance conservation phase phase-space canonical-transformation observable observables wavefunction
matrix-algebra vector-space inner-product outer-product cross-product dot-product projection projections rotation rotations translation translations
numerical numerical-analysis error errors approximation interpolation extrapolation quadrature finite-difference finite-element monte-carlo
fractal fractals self-similar dimension-fractal measure-zero nowhere-dense dense residual genericity
game-theory game games strategy strategies equilibrium nash payoff utility auction voting social-choice
cryptography cipher hash entropy-source prime-field elliptic-curve rsa discrete-log zero-knowledge
information-theory coding codeword channel capacity shannon compression redundancy
category-theory homological-algebra algebraic-topology differential-geometry algebraic-geometry complex-analysis real-analysis measure-theory
linear-algebra abstract-algebra probability-theory statistics-theory set-theoretic categorical functorial canonicalization
"""

_CONVERSATIONAL_SOURCE = """
buzz bzz bzzz fly flies wing wings fruit banana brain neuron neurons synapse synapses insect tiny
human humans lol lmao wtf hmm huh yep nope nah yeah hey yo ok okay wow oops bruh pls please
honestly seriously weird cursed based cringe wild sus neat cool funny dumb smart stupid beautiful ugly
love hate vibe vibes mood chaos random sigma epsilon math maths mathematician mathematicians nerd nerds
proof theorem topology integral category algebra geometry analysis probability statistics physics code coding
compute computer bug bugs feature theoremcore proofcore flybrain flymath drosophila swarm buzzing
apparently literally basically actually imagine suppose consider behold wait look listen question answer
why what how who where when maybe perhaps surely obviously trivially canonically naturally generically
almost never always again still anyway meanwhile therefore hence thus indeed somehow whatever
today tonight morning evening coffee sleep sleepy awake hungry food sugar fruitfly lab experiment
post tweet thread reply like likes repost reposts bookmark bookmarks views followers internet timeline
antenna antennae proboscis larva maggot
"""

MEME_TOKENS = frozenset(
    {
        "buzz", "bzz", "bzzz", "fly", "flies", "wing", "wings", "fruit", "banana", "brain",
        "neuron", "neurons", "synapse", "synapses", "insect", "lol", "lmao", "wtf", "hmm", "huh",
        "nope", "bruh", "cursed", "based", "cringe", "sus", "flybrain", "flymath", "drosophila",
        "swarm", "buzzing", "fruitfly", "antenna", "antennae", "proboscis", "larva", "maggot",
    }
)


def _unique_words(source: str, *, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    result: list[str] = []
    seen = set(excluded)
    for token in source.split():
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


@dataclass(frozen=True, slots=True)
class FlyVocabulary:
    tokens: tuple[str, ...]
    meme_tokens: frozenset[str] = MEME_TOKENS
    _token_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("vocabulary tokens must be unique")
        if not self.meme_tokens <= set(self.tokens):
            raise ValueError("meme tokens must be a vocabulary subset")
        object.__setattr__(
            self,
            "_token_to_id",
            MappingProxyType({token: index for index, token in enumerate(self.tokens)}),
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def id_for(self, token: str) -> int:
        return self._token_to_id[token]

    def token_for(self, token_id: int) -> str:
        return self.tokens[token_id]


def build_v1_vocabulary() -> FlyVocabulary:
    general = _unique_words(
        "see " + _GENERAL_SOURCE,
        excluded=set(SPECIAL_TOKENS) | set(SYMBOL_TOKENS),
    )[:320]
    if len(general) != 320:
        raise RuntimeError(f"expected 320 general tokens, got {len(general)}")

    math = _unique_words(
        _MATH_SOURCE,
        excluded=set(SPECIAL_TOKENS) | set(SYMBOL_TOKENS) | set(general),
    )[:480]
    if len(math) != 480:
        raise RuntimeError(f"expected 480 math tokens, got {len(math)}")

    conversational = _unique_words(
        _CONVERSATIONAL_SOURCE,
        excluded=set(SPECIAL_TOKENS) | set(SYMBOL_TOKENS) | set(general) | set(math),
    )
    if len(conversational) != 128:
        raise RuntimeError(f"expected 128 conversational tokens, got {len(conversational)}")

    tokens = SPECIAL_TOKENS + SYMBOL_TOKENS + tuple(general) + tuple(math) + tuple(conversational)
    if len(tokens) != 1024:
        raise RuntimeError(f"expected 1024 total tokens, got {len(tokens)}")
    return FlyVocabulary(tokens=tokens)
