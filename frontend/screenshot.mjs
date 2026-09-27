// Opens the dashboard in headless Chrome, waits for live WebSocket data, saves a screenshot.
import puppeteer from 'puppeteer-core'
const url = process.argv[2] ?? 'http://localhost:3001/'
const out = process.argv[3] ?? '../logs/dashboard_live.png'
const tab = process.argv[4]
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--disable-gpu'] })
const page = await browser.newPage()
await page.setViewport({ width: 1680, height: 980 })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
await page.goto(url, { waitUntil: 'load', timeout: 90000 })
if (tab) { await page.evaluate((t) => { for (const el of document.querySelectorAll('.tab')) if (el.textContent === t) el.click() }, tab) }
await new Promise((r) => setTimeout(r, 6000))
const text = await page.evaluate(() => document.body.innerText)
const imgs = await page.evaluate(() => Array.from(document.images).filter((i) => i.src.startsWith('data:image/jpeg')).length)
await page.screenshot({ path: out })
console.log(JSON.stringify({ jpeg_images: imgs, has_waiting: text.includes('WAITING FOR FRAMES'), sim_live: /SIM (POLICY RUNNING|IDLE|TELEOP|REPLAY|EVALUATING|SCRIPTED DEMO|LIVE)/.test(text), train_state: (text.match(/TRAIN (\w+)/) || [])[1], device: (text.match(/DEVICE: (\w+)/) || [])[1], errors: errors.slice(0, 5) }))
await browser.close()
