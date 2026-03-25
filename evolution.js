/* ============================================
   EvolutionAI — Core Simulation Engine
   ============================================
   Models: natural selection, mutation, genetic
   drift, speciation, and adaptation through a
   population of organisms in a 2D environment.
   ============================================ */

(() => {
"use strict";

// ── Utilities ──────────────────────────────
const rand  = (a, b) => Math.random() * (b - a) + a;
const randInt = (a, b) => Math.floor(rand(a, b + 1));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp  = (a, b, t) => a + (b - a) * t;
const dist  = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const hsl   = (h, s, l) => `hsl(${h},${s}%,${l}%)`;

// ── Genome ─────────────────────────────────
// Each organism has a genome encoding 6 traits (0–1 range)
const TRAITS = ['size', 'speed', 'camouflage', 'metabolism', 'coldResist', 'heatResist'];

function randomGenome() {
    const g = {};
    for (const t of TRAITS) g[t] = rand(0.2, 0.8);
    return g;
}

function mutateGenome(genome, rate) {
    const g = { ...genome };
    for (const t of TRAITS) {
        if (Math.random() < rate) {
            g[t] = clamp(g[t] + rand(-0.15, 0.15), 0, 1);
        }
    }
    return g;
}

function crossover(a, b) {
    const g = {};
    for (const t of TRAITS) g[t] = Math.random() < 0.5 ? a[t] : b[t];
    return g;
}

function genomeDist(a, b) {
    let sum = 0;
    for (const t of TRAITS) sum += (a[t] - b[t]) ** 2;
    return Math.sqrt(sum / TRAITS.length);
}

// ── Species Tracking ───────────────────────
let speciesIdCounter = 0;

class Species {
    constructor(genome, parent = null) {
        this.id = speciesIdCounter++;
        this.name = this.generateName();
        this.hue = (this.id * 67 + 120) % 360;
        this.refGenome = { ...genome };
        this.parent = parent;
        this.bornGen = 0;
        this.extinctGen = null;
        this.peakPop = 0;
    }

    generateName() {
        const pre = ['Neo','Proto','Para','Meta','Micro','Macro','Pseudo','Ultra','Xeno','Arch'];
        const mid = ['morph','theri','saur','pod','branch','plast','gen','phylo','zo','bio'];
        const suf = ['us','is','um','ax','on','ix','or','ia','ex','oid'];
        return pre[this.id % pre.length] + mid[(this.id * 3) % mid.length] + suf[(this.id * 7) % suf.length];
    }

    get alive() { return this.extinctGen === null; }
}

// ── Organism ───────────────────────────────
class Organism {
    constructor(x, y, genome, species) {
        this.x = x;
        this.y = y;
        this.genome = genome;
        this.species = species;
        this.energy = 50 + rand(-10, 10);
        this.age = 0;
        this.maxAge = 80 + rand(-20, 20);
        this.alive = true;

        // Visual
        this.vx = rand(-1, 1);
        this.vy = rand(-1, 1);
        this.targetX = x;
        this.targetY = y;
    }

    get radius() { return 3 + this.genome.size * 6; }

    fitness(env) {
        let f = 0.5;

        // Temperature fitness
        const tempNorm = env.temperature / 100;
        if (tempNorm < 0.3) f += this.genome.coldResist * 0.3;
        else if (tempNorm > 0.7) f += this.genome.heatResist * 0.3;
        else f += 0.15;

        // Food: larger size + higher metabolism = more food needed
        const foodNeed = (this.genome.size * 0.5 + this.genome.metabolism * 0.5);
        const foodAvail = env.food / 100;
        f += (foodAvail - foodNeed * 0.6) * 0.3;

        // Predation: speed + camouflage help survive
        const predSurvival = (this.genome.speed * 0.5 + this.genome.camouflage * 0.5);
        const predPressure = env.predation / 100;
        f += (predSurvival - predPressure * 0.5) * 0.3;

        // Terrain bonuses
        switch (env.terrain) {
            case 'forest':   f += this.genome.camouflage * 0.15; break;
            case 'desert':   f += this.genome.heatResist * 0.1 - this.genome.size * 0.05; break;
            case 'arctic':   f += this.genome.coldResist * 0.15; break;
            case 'ocean':    f += this.genome.speed * 0.1 + (1 - this.genome.size) * 0.05; break;
        }

        return clamp(f, 0.05, 1);
    }
}

// ── Food Particle ──────────────────────────
class Food {
    constructor(x, y, w, h) {
        this.x = x || rand(10, w - 10);
        this.y = y || rand(10, h - 10);
        this.energy = rand(10, 30);
        this.eaten = false;
    }
}

// ── Simulation ─────────────────────────────
class Simulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.running = false;
        this.generation = 0;
        this.tick = 0;
        this.ticksPerGen = 200;
        this.speed = 1;

        this.organisms = [];
        this.food = [];
        this.species = [];
        this.allSpecies = [];

        this.env = {
            temperature: 50,
            food: 60,
            predation: 30,
            mutationRate: 0.05,
            terrain: 'plains'
        };

        this.activeEvent = null;
        this.eventTimer = 0;

        // History for charts
        this.history = {
            population: [],
            speciesCounts: [],
            avgTraits: [],
            avgFitness: []
        };

        // Theory detection state
        this.theoryState = {
            selection: 0, mutation: 0, drift: 0,
            speciation: 0, adaptation: 0
        };

        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.w = this.canvas.width;
        this.h = this.canvas.height;
    }

    init(popSize = 60) {
        this.organisms = [];
        this.food = [];
        this.species = [];
        this.allSpecies = [];
        this.generation = 0;
        this.tick = 0;
        this.history = { population: [], speciesCounts: [], avgTraits: [], avgFitness: [] };
        speciesIdCounter = 0;

        // Founder species
        const founderGenome = randomGenome();
        const sp = new Species(founderGenome);
        sp.bornGen = 0;
        this.species.push(sp);
        this.allSpecies.push(sp);

        for (let i = 0; i < popSize; i++) {
            const g = mutateGenome(founderGenome, 0.1);
            const org = new Organism(rand(40, this.w - 40), rand(40, this.h - 40), g, sp);
            this.organisms.push(org);
        }

        this.spawnFood();
        this.recordHistory();
        logEvent(this.generation, `Simulation started with ${popSize} organisms`, '');
    }

    spawnFood() {
        const count = Math.floor(this.env.food * 0.8 + 10);
        this.food = [];
        for (let i = 0; i < count; i++) {
            this.food.push(new Food(null, null, this.w, this.h));
        }
    }

    // ── Main Update ───────────────────────
    update() {
        if (!this.running) return;

        for (let s = 0; s < this.speed; s++) {
            this.tick++;
            this.updateEvent();
            this.moveOrganisms();
            this.feedOrganisms();
            this.applyEnvironment();
            this.removeDeadOrganisms();

            if (this.tick % this.ticksPerGen === 0) {
                this.nextGeneration();
            }

            // Replenish some food each tick
            if (this.tick % 10 === 0 && this.food.filter(f => !f.eaten).length < this.env.food * 0.3) {
                for (let i = 0; i < 5; i++) this.food.push(new Food(null, null, this.w, this.h));
            }
        }
    }

    moveOrganisms() {
        for (const org of this.organisms) {
            if (!org.alive) continue;

            // Seek nearest food
            let nearestFood = null;
            let nearestDist = Infinity;
            for (const f of this.food) {
                if (f.eaten) continue;
                const d = dist(org, f);
                if (d < nearestDist) { nearestDist = d; nearestFood = f; }
            }

            const spd = 0.5 + org.genome.speed * 2;

            if (nearestFood && nearestDist < 200) {
                const dx = nearestFood.x - org.x;
                const dy = nearestFood.y - org.y;
                const len = Math.hypot(dx, dy) || 1;
                org.vx = lerp(org.vx, dx / len * spd, 0.1);
                org.vy = lerp(org.vy, dy / len * spd, 0.1);
            } else {
                // Wander
                if (Math.random() < 0.02) {
                    org.vx += rand(-0.5, 0.5);
                    org.vy += rand(-0.5, 0.5);
                }
            }

            org.x += org.vx;
            org.y += org.vy;

            // Bounds
            if (org.x < 5 || org.x > this.w - 5) org.vx *= -1;
            if (org.y < 5 || org.y > this.h - 5) org.vy *= -1;
            org.x = clamp(org.x, 5, this.w - 5);
            org.y = clamp(org.y, 5, this.h - 5);

            // Energy cost per tick
            const moveCost = 0.02 + org.genome.metabolism * 0.04 + org.genome.speed * 0.02;
            org.energy -= moveCost;
            org.age += 0.01;
        }
    }

    feedOrganisms() {
        for (const org of this.organisms) {
            if (!org.alive) continue;
            for (const f of this.food) {
                if (f.eaten) continue;
                if (dist(org, f) < org.radius + 4) {
                    f.eaten = true;
                    org.energy += f.energy * (0.5 + org.genome.metabolism * 0.5);
                }
            }
        }
    }

    applyEnvironment() {
        const tempNorm = this.env.temperature / 100;
        const predRate = this.env.predation / 100;

        for (const org of this.organisms) {
            if (!org.alive) continue;

            // Temperature stress
            if (tempNorm < 0.2) org.energy -= (0.2 - tempNorm) * (1 - org.genome.coldResist) * 0.3;
            if (tempNorm > 0.8) org.energy -= (tempNorm - 0.8) * (1 - org.genome.heatResist) * 0.3;

            // Predation
            if (Math.random() < predRate * 0.003 * (1 - org.genome.camouflage * 0.6 - org.genome.speed * 0.3)) {
                org.energy -= 30;
            }

            // Death
            if (org.energy <= 0 || org.age > org.maxAge) {
                org.alive = false;
            }
        }
    }

    removeDeadOrganisms() {
        this.organisms = this.organisms.filter(o => o.alive);
    }

    // ── Generation Cycle ──────────────────
    nextGeneration() {
        this.generation++;

        if (this.organisms.length === 0) {
            logEvent(this.generation, 'Total extinction! All organisms perished.', 'event-extinction');
            this.running = false;
            updateButtons(false);
            return;
        }

        // Calculate fitness
        const fitnesses = this.organisms.map(o => o.fitness(this.env));
        const totalFit = fitnesses.reduce((a, b) => a + b, 0);

        // ── Natural Selection: select parents weighted by fitness ──
        const selectParent = () => {
            let r = Math.random() * totalFit;
            for (let i = 0; i < this.organisms.length; i++) {
                r -= fitnesses[i];
                if (r <= 0) return this.organisms[i];
            }
            return this.organisms[this.organisms.length - 1];
        };

        // Target population with carrying capacity
        const capacity = 40 + this.env.food * 0.8;
        const targetPop = Math.min(Math.max(this.organisms.length, 20), capacity);

        const newOrganisms = [];

        // Detect selection pressure
        const fitVariance = this.variance(fitnesses);
        if (fitVariance > 0.02) this.theoryState.selection = 1;
        else this.theoryState.selection *= 0.9;

        for (let i = 0; i < targetPop; i++) {
            const p1 = selectParent();
            const p2 = selectParent();
            let childGenome = crossover(p1.genome, p2.genome);

            // ── Mutation ──
            const mutated = mutateGenome(childGenome, this.env.mutationRate);
            if (genomeDist(childGenome, mutated) > 0.05) {
                this.theoryState.mutation = 1;
            }
            childGenome = mutated;

            // ── Species assignment / Speciation ──
            let assignedSpecies = p1.species;
            const distToParent = genomeDist(childGenome, assignedSpecies.refGenome);

            if (distToParent > 0.35) {
                // Diverged enough → new species!
                const newSp = new Species(childGenome, assignedSpecies);
                newSp.bornGen = this.generation;
                this.species.push(newSp);
                this.allSpecies.push(newSp);
                assignedSpecies = newSp;
                this.theoryState.speciation = 1;
                logEvent(this.generation, `New species: ${newSp.name} (from ${p1.species.name})`, 'event-speciation');
            }

            const child = new Organism(
                p1.x + rand(-20, 20),
                p1.y + rand(-20, 20),
                childGenome,
                assignedSpecies
            );
            child.x = clamp(child.x, 10, this.w - 10);
            child.y = clamp(child.y, 10, this.h - 10);
            newOrganisms.push(child);
        }

        this.organisms = newOrganisms;

        // ── Genetic Drift detection (small populations) ──
        if (this.organisms.length < 25) {
            this.theoryState.drift = 1;
        } else {
            this.theoryState.drift *= 0.85;
        }

        // ── Adaptation detection ──
        if (this.history.avgFitness.length > 3) {
            const recent = this.history.avgFitness.slice(-4);
            const trend = recent[3] - recent[0];
            if (trend > 0.03) this.theoryState.adaptation = 1;
            else this.theoryState.adaptation *= 0.9;
        }

        // Check for extinct species
        for (const sp of this.species) {
            const count = this.organisms.filter(o => o.species === sp).length;
            sp.peakPop = Math.max(sp.peakPop, count);
            if (count === 0 && sp.alive) {
                sp.extinctGen = this.generation;
                logEvent(this.generation, `${sp.name} went extinct`, 'event-extinction');
            }
        }
        this.species = this.species.filter(sp => sp.alive);

        // Replenish food
        this.spawnFood();

        // Record history
        this.recordHistory();
    }

    recordHistory() {
        const h = this.history;
        h.population.push(this.organisms.length);
        h.speciesCounts.push(this.species.length);

        const avg = {};
        for (const t of TRAITS) {
            avg[t] = this.organisms.reduce((s, o) => s + o.genome[t], 0) / (this.organisms.length || 1);
        }
        h.avgTraits.push(avg);

        const avgFit = this.organisms.reduce((s, o) => s + o.fitness(this.env), 0) / (this.organisms.length || 1);
        h.avgFitness.push(avgFit);
    }

    variance(arr) {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    }

    // ── Events ────────────────────────────
    triggerEvent(type) {
        this.activeEvent = type;
        this.eventTimer = 60; // ticks

        switch (type) {
            case 'meteor':
                // Kill 40-60% of population randomly
                const killRate = rand(0.4, 0.6);
                let killed = 0;
                for (const o of this.organisms) {
                    if (Math.random() < killRate) { o.alive = false; killed++; }
                }
                this.removeDeadOrganisms();
                logEvent(this.generation, `Meteor strike! ${killed} organisms killed`, 'event-catastrophe');
                this.theoryState.drift = 1;
                break;

            case 'ice-age':
                this.env.temperature = 5;
                document.getElementById('ctrl-temperature').value = 5;
                document.getElementById('val-temp').textContent = '5';
                logEvent(this.generation, 'Ice Age! Temperature dropped drastically', 'event-catastrophe');
                break;

            case 'plague':
                // Targets dense clusters
                for (const o of this.organisms) {
                    const nearby = this.organisms.filter(n => n !== o && dist(o, n) < 30).length;
                    if (nearby > 3 && Math.random() < 0.5) o.alive = false;
                }
                this.removeDeadOrganisms();
                logEvent(this.generation, `Plague! Dense populations hit hard`, 'event-catastrophe');
                break;

            case 'radiation':
                // Massive mutation burst
                for (const o of this.organisms) {
                    o.genome = mutateGenome(o.genome, 0.5);
                }
                this.theoryState.mutation = 1;
                logEvent(this.generation, 'Radiation burst! Massive mutations across population', 'event-mutation');
                break;
        }

        // Flash effect
        showEventFlash(type);
    }

    updateEvent() {
        if (this.activeEvent) {
            this.eventTimer--;
            if (this.eventTimer <= 0) {
                this.activeEvent = null;
                document.querySelectorAll('.btn-event').forEach(b => b.classList.remove('active-event'));
            }
        }
    }

    // ── Render ─────────────────────────────
    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.w, this.h);

        // Background based on terrain
        this.renderBackground(ctx);

        // Food
        for (const f of this.food) {
            if (f.eaten) continue;
            ctx.beginPath();
            ctx.arc(f.x, f.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(74, 222, 128, 0.5)';
            ctx.fill();
        }

        // Organisms
        for (const org of this.organisms) {
            const r = org.radius;
            const sp = org.species;
            const alpha = 0.4 + org.genome.camouflage * 0.6;

            ctx.beginPath();
            ctx.arc(org.x, org.y, r, 0, Math.PI * 2);
            ctx.fillStyle = hsl(sp.hue, 70, 50 + org.genome.size * 20);
            ctx.globalAlpha = alpha;
            ctx.fill();
            ctx.globalAlpha = 1;

            // Outline
            ctx.strokeStyle = hsl(sp.hue, 80, 70);
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Speed indicator (trail)
            if (org.genome.speed > 0.6) {
                ctx.beginPath();
                ctx.moveTo(org.x - org.vx * 4, org.y - org.vy * 4);
                ctx.lineTo(org.x - org.vx * 8, org.y - org.vy * 8);
                ctx.strokeStyle = hsl(sp.hue, 50, 50);
                ctx.globalAlpha = 0.3;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        }

        // Event overlay
        if (this.activeEvent) {
            ctx.fillStyle = this.getEventOverlayColor();
            ctx.globalAlpha = 0.08 + Math.sin(this.tick * 0.1) * 0.04;
            ctx.fillRect(0, 0, this.w, this.h);
            ctx.globalAlpha = 1;
        }
    }

    renderBackground(ctx) {
        const colors = {
            plains:  ['#0d1117', '#111d13'],
            forest:  ['#0a1510', '#0d1f14'],
            desert:  ['#1a1408', '#211a0b'],
            arctic:  ['#0e1520', '#121d2e'],
            ocean:   ['#060e1a', '#091425']
        };
        const [c1, c2] = colors[this.env.terrain] || colors.plains;
        const grad = ctx.createLinearGradient(0, 0, 0, this.h);
        grad.addColorStop(0, c1);
        grad.addColorStop(1, c2);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, this.w, this.h);

        // Subtle grid
        ctx.strokeStyle = 'rgba(255,255,255,0.015)';
        ctx.lineWidth = 1;
        for (let x = 0; x < this.w; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.h); ctx.stroke();
        }
        for (let y = 0; y < this.h; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.w, y); ctx.stroke();
        }
    }

    getEventOverlayColor() {
        switch (this.activeEvent) {
            case 'meteor':    return '#f87171';
            case 'ice-age':   return '#93c5fd';
            case 'plague':    return '#4ade80';
            case 'radiation': return '#a78bfa';
            default: return '#fff';
        }
    }
}

// ── Charts ─────────────────────────────────
class MiniChart {
    constructor(canvasId, opts = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.opts = opts;
        this.resize();
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width - 32;
        this.w = this.canvas.width;
        this.h = this.canvas.height;
    }

    drawLine(data, color, maxVal = null) {
        if (data.length < 2) return;
        const ctx = this.ctx;
        const max = maxVal || Math.max(...data, 1);
        const step = this.w / (data.length - 1);

        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = i * step;
            const y = this.h - (data[i] / max) * (this.h - 10) - 5;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Fill under
        ctx.lineTo((data.length - 1) * step, this.h);
        ctx.lineTo(0, this.h);
        ctx.closePath();
        ctx.fillStyle = color.replace(')', ', 0.1)').replace('hsl', 'hsla').replace('rgb', 'rgba');
        ctx.fill();
    }

    clear() {
        this.ctx.clearRect(0, 0, this.w, this.h);
        this.ctx.fillStyle = '#1e293b';
        this.ctx.fillRect(0, 0, this.w, this.h);

        // Axis lines
        this.ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        this.ctx.lineWidth = 1;
        for (let y = 0; y < this.h; y += this.h / 4) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.w, y);
            this.ctx.stroke();
        }
    }
}

class BarChart extends MiniChart {
    drawBars(values, colors, labels) {
        const barW = (this.w - 20) / values.length;
        const max = Math.max(...values, 0.01);

        for (let i = 0; i < values.length; i++) {
            const barH = (values[i] / max) * (this.h - 25);
            const x = 10 + i * barW;
            const y = this.h - barH - 5;

            this.ctx.fillStyle = colors[i] || '#4ade80';
            this.ctx.fillRect(x + 2, y, barW - 4, barH);

            // Label
            this.ctx.fillStyle = '#94a3b8';
            this.ctx.font = '9px Inter';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(labels[i].slice(0, 4), x + barW / 2, this.h - 1);

            // Value
            this.ctx.fillStyle = '#e2e8f0';
            this.ctx.font = '9px JetBrains Mono';
            this.ctx.fillText(values[i].toFixed(2), x + barW / 2, y - 3);
        }
    }
}

// ── Phylogenetic Tree Renderer ─────────────
function renderPhyloTree(allSpecies, container) {
    container.innerHTML = '';

    // Build tree
    const roots = allSpecies.filter(sp => !sp.parent);
    function renderNode(sp, depth) {
        const div = document.createElement('div');
        div.className = 'phylo-node' + (sp.alive ? '' : ' extinct');
        div.style.paddingLeft = (depth * 14) + 'px';

        const dot = document.createElement('span');
        dot.className = 'species-dot';
        dot.style.background = hsl(sp.hue, 70, 55);
        div.appendChild(dot);

        const label = document.createTextNode(
            `${sp.name} (Gen ${sp.bornGen}${sp.extinctGen ? '–' + sp.extinctGen : ''})`
        );
        div.appendChild(label);
        container.appendChild(div);

        const children = allSpecies.filter(s => s.parent === sp);
        for (const child of children) renderNode(child, depth + 1);
    }

    for (const root of roots) renderNode(root, 0);
}

// ── Event Log ──────────────────────────────
function logEvent(gen, msg, cls = '') {
    const log = document.getElementById('event-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + cls;
    entry.innerHTML = `<span class="gen-tag">G${gen}</span>${msg}`;
    log.prepend(entry);

    // Limit entries
    while (log.children.length > 100) log.removeChild(log.lastChild);
}

// ── Event Flash ────────────────────────────
function showEventFlash(type) {
    // Remove existing flashes
    document.querySelectorAll('.event-flash').forEach(el => el.remove());

    const flash = document.createElement('div');
    flash.className = `event-flash ${type}`;
    document.querySelector('.canvas-wrapper').appendChild(flash);

    requestAnimationFrame(() => {
        flash.classList.add('show');
        setTimeout(() => {
            flash.classList.remove('show');
            setTimeout(() => flash.remove(), 500);
        }, 800);
    });
}

// ── UI Wiring ──────────────────────────────
function updateButtons(running) {
    document.getElementById('btn-start').disabled = running;
    document.getElementById('btn-pause').disabled = !running;
}

function updateTheoryCards(state) {
    const cards = document.querySelectorAll('.card');
    const theories = ['selection', 'mutation', 'drift', 'speciation', 'adaptation'];
    const hlClasses = ['highlight-selection', 'highlight-mutation', 'highlight-drift', 'highlight-speciation', 'highlight-adaptation'];

    cards.forEach((card, i) => {
        const key = theories[i];
        const val = state[key];
        const hl = hlClasses[i];

        card.classList.remove('active', ...hlClasses);
        if (val > 0.5) {
            card.classList.add(hl);
            const ind = card.querySelector('.card-indicator');
            ind.classList.add('active');
            ind.style.background = getComputedStyle(document.documentElement).getPropertyValue(
                key === 'selection' ? '--accent' :
                key === 'mutation' ? '--purple' :
                key === 'drift' ? '--warning' :
                key === 'speciation' ? '--accent2' : '--accent'
            );
        } else {
            const ind = card.querySelector('.card-indicator');
            ind.classList.remove('active');
            ind.style.background = '';
        }
    });
}

// ── Main Init ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('world-canvas');
    const sim = new Simulation(canvas);

    const popChart = new MiniChart('chart-population');
    const traitChart = new BarChart('chart-traits');
    const fitnessChart = new MiniChart('chart-fitness');

    // Controls
    const controls = {
        temperature: document.getElementById('ctrl-temperature'),
        food: document.getElementById('ctrl-food'),
        predation: document.getElementById('ctrl-predation'),
        mutation: document.getElementById('ctrl-mutation'),
        speed: document.getElementById('ctrl-speed'),
        terrain: document.getElementById('ctrl-terrain'),
    };

    const vals = {
        temperature: document.getElementById('val-temp'),
        food: document.getElementById('val-food'),
        predation: document.getElementById('val-predation'),
        mutation: document.getElementById('val-mutation'),
        speed: document.getElementById('val-speed'),
    };

    for (const [key, el] of Object.entries(controls)) {
        if (key === 'terrain') {
            el.addEventListener('change', () => { sim.env.terrain = el.value; });
            continue;
        }
        el.addEventListener('input', () => {
            const v = parseFloat(el.value);
            vals[key].textContent = v;
            if (key === 'speed') { sim.speed = v; return; }
            if (key === 'mutation') { sim.env.mutationRate = v / 100; return; }
            sim.env[key] = v;
        });
    }

    // Buttons
    document.getElementById('btn-start').addEventListener('click', () => {
        if (sim.organisms.length === 0) sim.init();
        sim.running = true;
        updateButtons(true);
    });

    document.getElementById('btn-pause').addEventListener('click', () => {
        sim.running = false;
        updateButtons(false);
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        sim.running = false;
        updateButtons(false);
        sim.init();
        document.getElementById('event-log').innerHTML = '';
    });

    // Events
    document.querySelectorAll('.btn-event').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!sim.running) return;
            btn.classList.add('active-event');
            sim.triggerEvent(btn.dataset.event);
        });
    });

    // ── Game Loop ──────────────────────────
    function loop() {
        sim.update();
        sim.render();

        // Update HUD
        document.getElementById('gen-counter').textContent = sim.generation;
        document.getElementById('pop-counter').textContent = sim.organisms.length;
        document.getElementById('species-counter').textContent = sim.species.length;

        // Update charts every few frames
        if (sim.tick % 5 === 0) {
            // Population chart
            popChart.clear();
            const maxPop = Math.max(...sim.history.population, 1);
            popChart.drawLine(sim.history.population, 'rgb(74, 222, 128)', maxPop * 1.2);
            if (sim.history.speciesCounts.length > 1) {
                popChart.drawLine(
                    sim.history.speciesCounts.map(v => v * maxPop / Math.max(...sim.history.speciesCounts, 1)),
                    'rgb(34, 211, 238)', maxPop * 1.2
                );
            }

            // Trait chart
            if (sim.history.avgTraits.length > 0) {
                traitChart.clear();
                const latest = sim.history.avgTraits[sim.history.avgTraits.length - 1];
                const tVals = TRAITS.map(t => latest[t]);
                const tColors = ['#4ade80', '#22d3ee', '#a78bfa', '#fbbf24', '#93c5fd', '#f87171'];
                traitChart.drawBars(tVals, tColors, TRAITS);
            }

            // Fitness chart
            fitnessChart.clear();
            fitnessChart.drawLine(sim.history.avgFitness, 'rgb(244, 114, 182)', 1);

            // Phylo tree
            renderPhyloTree(sim.allSpecies, document.getElementById('phylo-tree'));

            // Theory cards
            updateTheoryCards(sim.theoryState);
        }

        requestAnimationFrame(loop);
    }

    // Start rendering (not simulating yet)
    sim.init();
    loop();
});

})();
