// Align Hebrew letters to IPA chunks from a TSV file.
// Input:  TSV with columns: hebrew\tipa
// Output: JSONL with format: {hebrew: [[char, chunk], ...]}
//
// Usage: ./align input.tsv output.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/dlclark/regexp2"
	"github.com/dustin/go-humanize"
)

var consonants = map[rune][]string{
	'א': {"ʔ", ""},
	'ב': {"b", "v"},
	'ג': {"ɡ", "dʒ"},
	'ד': {"d"},
	'ה': {"h", ""},
	'ו': {"v", "w", ""},
	'ז': {"z", "ʒ"},
	'ח': {"χ"},
	'ט': {"t"},
	'י': {"j", ""},
	'כ': {"k", "χ"},
	'ך': {"k", "χ"},
	'ל': {"l"},
	'מ': {"m"},
	'ם': {"m"},
	'נ': {"n"},
	'ן': {"n"},
	'ס': {"s"},
	'ע': {"ʔ", ""},
	'פ': {"p", "f"},
	'ף': {"p", "f"},
	'צ': {"ts", "tʃ"},
	'ץ': {"ts", "tʃ"},
	'ק': {"k"},
	'ר': {"ʁ"},
	'ש': {"ʃ", "s", ""},
	'ת': {"t"},
}

var vowels = []string{"a", "e", "i", "o", "u"}

var (
	nikudRe  = regexp2.MustCompile(`[\p{M}|]`, 0)
	hebRe    = regexp.MustCompile(`[^\x{05D0}-\x{05EA}]`)
	ipaRe    = regexp.MustCompile(`[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]`)
	wordCache sync.Map // heb word string → *regexp.Regexp
)

// nonCapturing wraps each pattern in a non-capturing group and joins with |
func nonCapturing(parts []string) string {
	wrapped := make([]string, len(parts))
	for i, p := range parts {
		wrapped[i] = "(?:" + p + ")"
	}
	return strings.Join(wrapped, "|")
}

// letterPattern returns a regex capture group for all IPA chunks a Hebrew letter can produce.
// Mirrors Python's _chunk_pattern.
func letterPattern(ch rune, allowed []string) string {
	vowelRe := "ˈ?(?:" + strings.Join(vowels, "|") + ")?"
	vowelsRe := "ˈ?(?:" + strings.Join(vowels, "|") + ")"

	// Split allowed consonants from silent marker
	hasSilent := false
	var cons []string
	for _, c := range allowed {
		if c == "" {
			hasSilent = true
		} else {
			cons = append(cons, regexp.QuoteMeta(c))
		}
	}
	sort.Slice(cons, func(i, j int) bool { return len(cons[i]) > len(cons[j]) })

	var parts []string

	// Consonant(s) + optional stress + optional vowel
	if len(cons) > 0 {
		consRe := cons[0]
		if len(cons) > 1 {
			consRe = "(?:" + strings.Join(cons, "|") + ")"
		}
		parts = append(parts, consRe+vowelRe)
	}

	// Special vowel-only cases
	if ch == 'ו' {
		parts = append(parts, "ˈ?(?:u|o)")
	}
	if ch == 'י' {
		parts = append(parts, "ˈ?i")
	}

	// ח furtive patah: optional vowel + χ (reversed order)
	if ch == 'ח' {
		parts = append(parts, vowelsRe+"?χ")
	}

	// Silent: vowel-only, stress-only, or empty
	if hasSilent {
		parts = append(parts, vowelsRe+"|ˈ|")
	}

	combined := nonCapturing(parts)
	if hasSilent || ch == 'ו' || ch == 'י' {
		combined += "|" // allow empty match
	}
	return "(" + combined + ")"
}

var letterPatterns = func() map[rune]string {
	m := make(map[rune]string, len(consonants))
	for ch, allowed := range consonants {
		m[ch] = letterPattern(ch, allowed)
	}
	return m
}()

func stripNikud(s string) string {
	r, _ := nikudRe.Replace(s, "", 0, -1)
	return r
}

func alignWord(heb, ipa string) [][2]string {
	chars := []rune(heb)

	var re *regexp.Regexp
	if v, ok := wordCache.Load(heb); ok {
		re = v.(*regexp.Regexp)
	} else {
		var pat strings.Builder
		for _, c := range chars {
			if p, ok := letterPatterns[c]; ok {
				pat.WriteString(p)
			} else {
				pat.WriteString(`(\S*)`)
			}
		}
		re = regexp.MustCompile("^(?:" + pat.String() + ")$")
		wordCache.Store(heb, re)
	}

	m := re.FindStringSubmatch(ipa)
	if m == nil {
		return nil
	}
	alignResult := make([][2]string, len(chars))
	for i, c := range chars {
		alignResult[i] = [2]string{string(c), m[i+1]}
	}
	return alignResult
}

func alignSentence(heb, ipa string) [][2]string {
	hebWords := strings.Fields(heb)
	ipaWords := strings.Fields(ipa)
	if len(hebWords) != len(ipaWords) {
		return nil
	}
	var out [][2]string
	for i, hw := range hebWords {
		hCore := hebRe.ReplaceAllString(hw, "")
		iCore := ipaRe.ReplaceAllString(ipaWords[i], "")
		if hCore == "" {
			continue
		}
		aligned := alignWord(hCore, iCore)
		if aligned == nil {
			return nil
		}
		if len(out) > 0 {
			out = append(out, [2]string{" ", " "})
		}
		out = append(out, aligned...)
	}
	return out
}

type alignResult struct {
	idx       int
	heb, ipa  string
	alignment [][2]string
	bytes     int
}

func readLines(path string) ([]string, int64) {
	f, _ := os.Open(path)
	defer f.Close()
	var lines []string
	var total int64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		lines = append(lines, line)
		total += int64(len(line)) + 1
	}
	return lines, total
}

func processLine(line string, idx int) alignResult {
	tab := strings.IndexByte(line, '\t')
	if tab < 0 {
		return alignResult{idx: idx, bytes: len(line) + 1}
	}
	heb := stripNikud(line[:tab])
	ipa := line[tab+1:]
	return alignResult{idx: idx, heb: heb, ipa: ipa, alignment: alignSentence(heb, ipa), bytes: len(line) + 1}
}

func runWorkers(lines []string) <-chan alignResult {
	nw := runtime.NumCPU()
	jobs := make(chan int, nw*4)
	out := make(chan alignResult, nw*4)
	var wg sync.WaitGroup
	for range nw {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				out <- processLine(lines[idx], idx)
			}
		}()
	}
	go func() {
		for i := range lines {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(out)
	}()
	return out
}

func writeResults(alignResults <-chan alignResult, totalBytes int64, outW, failW *bufio.Writer) {
	var total, aligned, failed, done int64
	lastPct := int64(-1)
	nextIdx := 0
	pending := map[int]alignResult{}

	for r := range alignResults {
		pending[r.idx] = r
		for {
			r, ok := pending[nextIdx]
			if !ok {
				break
			}
			delete(pending, nextIdx)
			nextIdx++
			done += int64(r.bytes)
			if pct := done * 100 / totalBytes; pct != lastPct {
				lastPct = pct
				fmt.Fprintf(os.Stderr, "\rAligning: %3d%% [%s / %s]", pct, humanize.Bytes(uint64(done)), humanize.Bytes(uint64(totalBytes)))
			}
			if r.heb == "" {
				continue
			}
			total++
			if r.alignment != nil {
				aligned++
				b, _ := json.Marshal(map[string]any{"hebrew": r.heb, "alignment": r.alignment})
				outW.Write(b)
				outW.WriteByte('\n')
			} else {
				failed++
				failW.WriteString(r.heb + "\t" + r.ipa + "\n")
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\rAligning: 100%% [%s / %s]\n", humanize.Bytes(uint64(totalBytes)), humanize.Bytes(uint64(totalBytes)))
	fmt.Fprintf(os.Stderr, "\nTotal:   %d\nAligned: %d (%.1f%%)\nFailed:  %d (%.1f%%)\n",
		total, aligned, float64(aligned)/float64(total)*100, failed, float64(failed)/float64(total)*100)
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s input.tsv output.jsonl\n", os.Args[0])
		os.Exit(1)
	}
	inputPath, outputPath := os.Args[1], os.Args[2]
	failPath := strings.TrimSuffix(outputPath, ".jsonl") + "_failures.txt"

	lines, totalBytes := readLines(inputPath)

	fout, _ := os.Create(outputPath)
	ffail, _ := os.Create(failPath)
	defer fout.Close()
	defer ffail.Close()
	outW := bufio.NewWriterSize(fout, 4<<20)
	failW := bufio.NewWriterSize(ffail, 1<<20)
	defer outW.Flush()
	defer failW.Flush()

	writeResults(runWorkers(lines), totalBytes, outW, failW)
}
