// Deterministic manuscript-specific Markdown -> TeX renderer (Node 22, no dependencies).
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const here=path.dirname(fileURLToPath(import.meta.url));
const PREAMBLE="\\documentclass[11pt,a4paper]{article}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n\\usepackage{amsmath,amssymb}\n\\usepackage[margin=26mm]{geometry}\n\\usepackage{microtype}\n\\usepackage{booktabs,tabularx,array}\n\\usepackage{tikz}\n\\usepackage{xurl}\n\\usepackage[hidelinks]{hyperref}\n\\usepackage{caption}\n\\captionsetup{font=small,labelfont=bf}\n\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{5pt plus 1pt}\n\\setlength{\\emergencystretch}{3em}\n\\raggedbottom\n\\newcolumntype{Y}{>{\\raggedright\\arraybackslash}X}\n\\title{";
function escapeTeX(s){
 return s.replace(/[\\{}%$&#_^~<>|]/g,c=>({'\\':'\\textbackslash{}','{':'\\{','}':'\\}','%':'\\%','$':'\\$','&':'\\&','#':'\\#','_':'\\_','^':'\\textasciicircum{}','~':'\\textasciitilde{}','<':'\\textless{}','>':'\\textgreater{}','|':'\\textbar{}'})[c]).replaceAll('–','--').replaceAll('•',' / ');
}
function inline(s){
 const tokens=[];
 const keep=t=>{tokens.push(t);return '\u0001'+(tokens.length-1)+'\u0002';};
 s=s.replace(/\$([^$\n]+)\$/g,(_,m)=>keep('$'+m+'$'));
 s=s.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,(_,t,u)=>keep('\\href{'+u+'}{'+escapeTeX(t)+'}'));
 s=s.replace(/\bhttps?:\/\/[^\s)]+/g,u=>keep('\\url{'+u.replace(/[.,;]$/,'')+'}')+(u.match(/[.,;]$/)?.[0]??''));
 s=s.replace(/`([^`]+)`/g,(_,t)=>keep('\\texttt{'+escapeTeX(t)+'}'));
 s=s.replace(/\[(\d+(?:,\d+)*)\]/g,(_,t)=>keep('\\cite{'+t.split(',').map(x=>'ref'+x).join(',')+'}'));
 s=s.replace(/\*\*([^*]+)\*\*/g,(_,t)=>keep('\\textbf{'+escapeTeX(t)+'}'));
 s=s.replace(/\*([^*]+)\*/g,(_,t)=>keep('\\emph{'+escapeTeX(t)+'}'));
 const math=[
 ['O:Theta -> Y','O:\\Theta\\to Y'],['g:Theta -> G','g:\\Theta\\to G'],['g = d composed with O','g=d\\circ O'],
 ['T=R/Z','\\mathbb T=\\mathbb R/\\mathbb Z'],['g=E-V+1','g=E-V+1'],['V-1','V-1'],['d_1,...,d_r','d_1,\\ldots,d_r'],['UAV=D','UAV=D'],['d_i x=0','d_i x=0'],
 ['(Z/8)^3','(\\mathbb Z/8)^3'],['2I','2I'],['x1+x2+x3','x_1+x_2+x_3'],['x1+x2','x_1+x_2'],['x2+x3','x_2+x_3'],['x1+x3','x_1+x_3'],
 ['|+000>','|+000\\rangle'],['|0>','|0\\rangle'],['|+> tensor |Phi+>','|+\\rangle\\otimes|\\Phi^+\\rangle'],['(P+<XXX>)/2','(P+\\langle XXX\\rangle)/2'],
 ['w(theta)=p(theta|D)','w(\\theta)=p(\\theta\\mid D)'],['p_a(y|theta)','p_a(y\\mid\\theta)'],['c_a>0','c_a>0'],['Z_a=f_a(theta)','Z_a=f_a(\\theta)'],['I(Z_a;Y_a|D)','I(Z_a;Y_a\\mid D)'],
 ['e_theta=TV(p_a(.|theta),q_a(.|theta))','e_\\theta=\\operatorname{TV}(p_a(\\cdot\\mid\\theta),q_a(\\cdot\\mid\\theta))'],['delta=sum_theta w_theta e_theta','\\delta=\\sum_\\theta w_\\theta e_\\theta'],
 ['f_K(t)=h_2(t)+t log(K-1)','f_K(t)=h_2(t)+t\\log(K-1)'],['t<=1-1/K','t\\le 1-1/K'],['log K','\\log K'],['alpha_probe+(-1)^b delta+b*pi','\\alpha_{\\rm probe}+(-1)^b\\delta+b\\pi'],
 ['P0=max(SE0^2,floor^2)','P_0=\\max(\\mathrm{SE}_0^2,\\mathrm{floor}^2)'],['R=max(SE1^2,floor^2)','R=\\max(\\mathrm{SE}_1^2,\\mathrm{floor}^2)'],['P-=P0+q^2 dt','P^-=P_0+q^2\\Delta t'],['mu0','\\mu_0'],['sqrt(hour)','\\sqrt{\\mathrm{hour}}']
 ];
 for(const [a,b] of math){if(s.includes(a))s=s.split(a).join(keep('$'+b+'$'));}
 s=s.replace(/\b[0-9]+(?:\.[0-9]+)?e-[0-9]+\b/g,t=>{const [a,b]=t.split('e-');return keep('$'+a+'\\times10^{-'+b+'}$');});
 return escapeTeX(s).replace(/\u0001(\d+)\u0002/g,(_,i)=>tokens[Number(i)]);
}
function table(rows,caption){
 const cells=rows.map(r=>r.trim().slice(1,-1).split('|').map(c=>c.trim()));
 const n=cells[0].length;
 if(cells.some(r=>r.length!==n))throw Error('inconsistent table width');
 const line=r=>r.map(x=>inline(x).replace(/,(?=[+-]?\d+\.\d+\])/g,',\\allowbreak ')).join(' & ')+' \\\\';
 const wide=n>=5;
 const spec=n===2?'@{}lY@{}':'@{}'+Array(n).fill('Y').join('')+'@{}';
 const inner=['\\centering',wide?'\\small':'\\normalsize',wide?'\\setlength{\\tabcolsep}{3pt}':'','\\begin{tabularx}{\\linewidth}{'+spec+'}','\\toprule',line(cells[0]),'\\midrule',...cells.slice(2).map(line),'\\bottomrule','\\end{tabularx}'].filter(Boolean).join('\n');
 return caption?'\\begin{table}[htbp]\n\\caption{'+inline(caption)+'}\n'+inner+'\n\\end{table}':'\\begin{center}\n'+inner+'\n\\end{center}';
}
function render(md){
 const lines=md.replace(/\r\n/g,'\n').split('\n');
 let result=[],inMath=false,inAbstract=false,inBibliography=false,caption=null,i=1;
 const title=lines[0].replace(/^# /,'');
 if(!/^# /.test(lines[0]))throw Error('missing title');
 result.push(PREAMBLE+escapeTeX(title)+'}', '\\author{Author information to be confirmed}', '\\date{Working manuscript, version 0.2\\\\28 November 2026}', '\\begin{document}', '\\maketitle');
 while(i<lines.length){
  let l=lines[i++];
  if(l==='$$'){result.push(inMath?'\\]':'\\[');inMath=!inMath;continue;}
  if(inMath){result.push(l);continue;}
  if(l.startsWith('**Research manuscript,')||l.startsWith('Author list and affiliations'))continue;
  if(l==='## Abstract'){result.push('\\begin{abstract}');inAbstract=true;continue;}
  if(l.startsWith('**Keywords:**')){if(inAbstract){result.push('\\end{abstract}');inAbstract=false;}result.push('\\smallskip\\noindent '+inline(l)+'\\par');continue;}
  if(l==='## References'){result.push('\\begin{thebibliography}{99}');inBibliography=true;continue;}
  if(l==='## Pre-submission checklist'){if(inBibliography){result.push('\\end{thebibliography}');inBibliography=false;}break;}
  if(inBibliography&&/^\d+\. /.test(l)){const m=l.match(/^(\d+)\. (.*)$/);result.push('\\bibitem{ref'+m[1]+'} '+inline(m[2]));continue;}
  if(/^#{2,3} /.test(l)){
   if(inAbstract){result.push('\\end{abstract}');inAbstract=false;}
   const m=l.match(/^(#{2,3}) (.*)$/),name=m[2].replace(/^\d+(?:\.\d+)?\.?\s+/,'');
   result.push('\\'+(m[1].length===2?'section':'subsection')+(m[1].length===2&&m[2].startsWith('Data,')?'*':'')+'{'+inline(name)+'}');continue;
  }
  if(/^\*\*Table \d+\.\*\*/.test(l)){caption=l.replace(/^\*\*Table \d+\.\*\*\s*/,'');continue;}
  if(l.startsWith('|')&&l.endsWith('|')){
   const t=[l];while(i<lines.length&&lines[i].startsWith('|')&&lines[i].endsWith('|'))t.push(lines[i++]);
   if(t.length<3)throw Error('malformed table');
   result.push(table(t,caption));caption=null;continue;
  }
  if(/^    \s*\S/.test(l)){
   const code=[l.slice(4)];while(i<lines.length&&/^    \s*\S/.test(lines[i]))code.push(lines[i++].slice(4));
   result.push('{\\small\\begin{verbatim}',...code,'\\end{verbatim}}');continue;
  }
  if(l.startsWith('- ')){const items=[l.slice(2)];while(i<lines.length&&lines[i].startsWith('- '))items.push(lines[i++].slice(2));result.push('\\begin{itemize}',...items.map(t=>'\\item '+inline(t)),'\\end{itemize}');continue;}
  if(l.startsWith('Qubit 23 supplies much of the observed benefit.'))result.push('\\input{results_figure.tex}');
  result.push(inline(l));
 }
 if(inAbstract)result.push('\\end{abstract}');
 if(inBibliography)result.push('\\end{thebibliography}');
 if(inMath||caption!==null)throw Error('unfinished math or table caption');
 result.push('\\end{document}','');
 return result.join('\n');
}
const md=fs.readFileSync(path.join(here,'PAPER.md'),'utf8');
fs.writeFileSync(path.join(here,'paper.tex'),render(md));
