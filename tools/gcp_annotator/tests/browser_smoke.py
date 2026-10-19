"""Optional real-image browser QA. All synthetic clicks live in a deleted temp store.

Requires playwright + chromium, outside the annotation runtime requirements.
Run from repository root; do not use the generated QA metrics as scientific GT.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from tools.gcp_annotator.core import Catalog, DEFAULT_ROOT, sha_file, write_json, now

REPO=Path(__file__).resolve().parents[3]


def run(args):
    from playwright.sync_api import sync_playwright
    report_dir=args.report_dir.resolve();report_dir.mkdir(parents=True,exist_ok=True)
    catalog=Catalog(args.data_root)
    originals={str(catalog.manifest):sha_file(catalog.manifest)}
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
        print('PASS',name,flush=True)
    qa_parent=args.data_root/'ground_gcp_annotations/qa'
    qa_parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='browser_',dir=qa_parent) as temporary:
        temporary=Path(temporary)
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        command=[sys.executable,'-m','tools.gcp_annotator.app','--data-root',str(args.data_root),'--output',str(temporary/'labels'),'--port',str(port)]
        log=(report_dir/'server.log').open('w')
        server=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        def ready():
            for _ in range(150):
                if server.poll() is not None:raise RuntimeError('Server exited; see server.log')
                try:
                    with urllib.request.urlopen(url+'/api/catalog',timeout=1):return
                except OSError:time.sleep(.1)
            raise TimeoutError('Server did not become ready')
        try:
            ready()
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
                context=browser.new_context(viewport={'width':1680,'height':1100},device_scale_factor=2)
                page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url);page.wait_for_function('doc && meta && !loading && views.A.image && views.B.image')
                pair=page.evaluate('doc.pair_id')
                for side in ('A','B'):
                    for mode in ('native','test'):
                        path=catalog.image_path(catalog.row(pair),side,mode);originals[str(path)]=sha_file(path)
                check('actual_googleearth_manifest_498_pairs',page.evaluate('catalog.total')==498)
                check('balanced_frozen_sample_20',page.evaluate('catalog.selection.pair_ids.length')==20)
                check('native_preview_default',page.locator('#mode').input_value()=='native')
                check('sync_v2_script_executed',page.locator('#viewSyncState').inner_text()=='双图联动 v2：缩放＋平移')
                check('script_URL_bypasses_previous_cached_version','?v=20261019-sync2' in page.evaluate("Array.from(document.scripts).find(s=>s.src.includes('/static/app.js')).src"))
                result=page.evaluate('''() => {
                  let max=0,zoom=0;
                  for(let i=0;i<1000;i++){
                    const p=[Math.sin(i)*500+500,Math.cos(i*.3)*500+500],v={x:i*.12-100,y:i*.4-200,scale:.05+i*.031};
                    const s=G.imageToScreen(p,v),back=G.screenToImage(s,v),z=G.zoomAt(v,s,1.23),fixed=G.screenToImage(s,z);
                    max=Math.max(max,...p.map((x,k)=>Math.abs(x-back[k])));
                    zoom=Math.max(zoom,...p.map((x,k)=>Math.abs(x-fixed[k])));
                  }
                  return {max,zoom,dpr:devicePixelRatio};
                }''')
                check('screen_roundtrip_subpixel_dpr2',result['max']<1e-9 and result['dpr']==2)
                check('cursor_centered_zoom_math',result['zoom']<1e-9)
                def click(side,point):
                    data=page.evaluate('''({side,p}) => {
                      const v=views[side],q=G.imageToScreen(toView(side,p),v.v),r=v.canvas.getBoundingClientRect();
                      v.canvas.addEventListener('pointerup',e=>{window.qaClick=[e.clientX-r.left,e.clientY-r.top];},{once:true});
                      return [r.left+q[0],r.top+q[1]];
                    }''',{'side':side,'p':point})
                    page.mouse.click(*data)
                    return page.evaluate('(side)=>fromView(side,G.screenToImage(window.qaClick,views[side].v))',side)
                click('B',[230.,240.]);check('B_cannot_create_orphan_point',page.evaluate('doc.points.length===0 && doc.draft===null'))
                expected_a=click('A',[245.125,261.375])
                page.wait_for_function('!dirty && !saving && doc.draft')
                page.reload();page.wait_for_function('doc && !loading && doc.draft')
                check('unfinished_A_draft_survives_reload',page.evaluate('doc.points.length===0 && !!doc.draft'))
                expected_b=click('B',[220.625,200.875]);page.wait_for_function('doc.points.length===1 && !dirty && !saving')
                stored=page.evaluate('doc.points[0]')
                check('A_B_pair_id_association',stored['id']=='GCP-001')
                check('native_to_test_click_coordinate_accuracy',max(abs(x-y) for x,y in zip(stored['a']+stored['b'],expected_a+expected_b))<1e-9)
                check('floating_point_coordinates_preserved',any(abs(x-round(x))>1e-5 for x in stored['a']+stored['b']))
                before_scales=page.evaluate('[views.A.v.scale,views.B.v.scale]')
                points_before_zoom=page.evaluate('JSON.stringify(doc.points)')
                box=page.locator('#canvasA').bounding_box();page.mouse.move(box['x']+box['width']*.4,box['y']+box['height']*.4)
                # Chromium may quantize wheel coordinates differently from a
                # preceding pointermove; measure the actual wheel event anchor.
                page.evaluate('''()=>views.A.canvas.addEventListener('wheel',e=>{
                  window.qaWheelScreen=views.A.screen(e);
                  window.qaWheelBefore=G.screenToImage(window.qaWheelScreen,views.A.v);
                  const [aw,ah]=views.A.size(),[bw,bh]=views.B.size();
                  window.qaPeerScreen=[window.qaWheelScreen[0]*bw/aw,window.qaWheelScreen[1]*bh/ah];
                  window.qaPeerBefore=G.screenToImage(window.qaPeerScreen,views.B.v);
                },{once:true,capture:true})''')
                page.mouse.wheel(0,-220);page.wait_for_timeout(100)
                before=page.evaluate('window.qaWheelBefore')
                after=page.evaluate('G.screenToImage(window.qaWheelScreen,views.A.v)')
                check('wheel_keeps_cursor_image_position',max(abs(x-y) for x,y in zip(before,after))<1e-9)
                scales=page.evaluate('[views.A.v.scale,views.B.v.scale]')
                check('A_wheel_zooms_both_views_equally',scales[0]>before_scales[0] and abs(scales[0]/before_scales[0]-scales[1]/before_scales[1])<1e-12)
                peer_after=page.evaluate('G.screenToImage(window.qaPeerScreen,views.B.v)')
                peer_before=page.evaluate('window.qaPeerBefore')
                check('peer_zoom_keeps_relative_screen_anchor',max(abs(x-y) for x,y in zip(peer_before,peer_after))<1e-9)
                box_b=page.locator('#canvasB').bounding_box();page.mouse.move(box_b['x']+box_b['width']*.6,box_b['y']+box_b['height']*.5)
                page.mouse.wheel(0,180);page.wait_for_timeout(100)
                after_b_zoom=page.evaluate('[views.A.v.scale,views.B.v.scale]')
                check('B_wheel_zooms_both_views_equally',after_b_zoom[1]<scales[1] and abs(after_b_zoom[0]/scales[0]-after_b_zoom[1]/scales[1])<1e-12)
                check('synchronized_zoom_never_changes_GCP_coordinates',page.evaluate('JSON.stringify(doc.points)')==points_before_zoom)
                page.evaluate('views.A.v.scale=95;views.B.v.scale=50')
                page.mouse.wheel(0,-100000);page.wait_for_timeout(100)
                upper=page.evaluate('[views.A.v.scale,views.B.v.scale]')
                page.mouse.wheel(0,-100000);page.wait_for_timeout(100)
                check('zoom_upper_limit_stops_both_views',upper[0]==100 and page.evaluate('[views.A.v.scale,views.B.v.scale]')==upper)
                page.mouse.wheel(0,100000);page.wait_for_timeout(100)
                lower=page.evaluate('[views.A.v.scale,views.B.v.scale]')
                page.mouse.wheel(0,100000);page.wait_for_timeout(100)
                check('zoom_lower_limit_stops_both_views',lower[1]==.01 and page.evaluate('[views.A.v.scale,views.B.v.scale]')==lower)
                page.locator('#fit').click()
                page.mouse.move(box['x']+box['width']*.4,box['y']+box['height']*.4)
                pan_before=page.evaluate('[{...views.A.v},{...views.B.v}]')
                page.keyboard.down('Space');page.mouse.down();page.mouse.move(box['x']+box['width']*.4+30,box['y']+box['height']*.4+20,steps=5);page.mouse.up();page.keyboard.up('Space')
                pan_after=page.evaluate('[{...views.A.v},{...views.B.v}]')
                check('A_space_drag_pans_both_views',all(abs(after['x']-before['x']-30)<1 and abs(after['y']-before['y']-20)<1 for before,after in zip(pan_before,pan_after)))
                page.mouse.move(box_b['x']+box_b['width']*.1,box_b['y']+box_b['height']*.1)
                pan_before=page.evaluate('[{...views.A.v},{...views.B.v}]')
                page.mouse.down();page.mouse.move(box_b['x']+box_b['width']*.1-24,box_b['y']+box_b['height']*.1+16,steps=5);page.mouse.up()
                pan_after=page.evaluate('[{...views.A.v},{...views.B.v}]')
                check('B_empty_area_drag_pans_both_views',all(abs(after['x']-before['x']+24)<1 and abs(after['y']-before['y']-16)<1 for before,after in zip(pan_before,pan_after)))
                pan_before=pan_after
                page.mouse.down(button='middle');page.mouse.move(box_b['x']+box_b['width']*.1-2,box_b['y']+box_b['height']*.1+2,steps=5);page.mouse.up(button='middle')
                pan_after=page.evaluate('[{...views.A.v},{...views.B.v}]')
                check('middle_drag_pans_both_views',all(abs(after['x']-before['x']-22)<1 and abs(after['y']-before['y']+14)<1 for before,after in zip(pan_before,pan_after)))
                check('synchronized_pan_never_changes_GCP_coordinates',page.evaluate('JSON.stringify(doc.points)')==points_before_zoom)
                page.locator('#fit').click()
                click('A',[320.25,305.75]);click('B',[300.75,290.125]);page.wait_for_function('doc.points.length===2 && !dirty && !saving')
                page.locator('tr[data-id="GCP-001"] td').first.click()
                unchanged_b=page.evaluate('doc.points[0].b.slice()')
                drag=page.evaluate('''()=>{const v=views.A,q=G.imageToScreen(toView('A',doc.points[0].a),v.v),r=v.canvas.getBoundingClientRect();return [r.left+q[0],r.top+q[1]];}''')
                old_a=page.evaluate('doc.points[0].a.slice()')
                views_before_point_drag=page.evaluate('JSON.stringify([views.A.v,views.B.v])')
                page.mouse.move(*drag);page.mouse.down();page.mouse.move(drag[0]+10,drag[1]+8,steps=8);page.mouse.up()
                check('drag_changes_only_selected_side',page.evaluate('doc.points[0].a')!=old_a and page.evaluate('doc.points[0].b')==unchanged_b)
                check('point_drag_does_not_pan_views',page.evaluate('JSON.stringify([views.A.v,views.B.v])')==views_before_point_drag)
                dragged=page.evaluate('doc.points[0].a.slice()');page.locator('#undo').click();check('drag_undo',page.evaluate('doc.points[0].a')==old_a)
                page.locator('#redo').click();check('drag_redo',page.evaluate('doc.points[0].a')==dragged)
                page.locator('tr[data-id="GCP-001"] td').first.click()
                page.locator('#delete').click();check('delete_entire_pair',page.evaluate('doc.points.length')==1)
                page.locator('#undo').click();check('delete_undo_restores_pair',page.evaluate('doc.points.length')==2)
                page.wait_for_function('!dirty && !saving')
                page.get_by_label('GCP-001 待复核',exact=True).check()
                page.wait_for_function('!dirty && !saving')
                check('review_flag_edit_after_autosave',page.evaluate('doc.points[0].needs_review'))
                page.get_by_label('GCP-001 待复核',exact=True).uncheck()
                page.wait_for_function('!dirty && !saving')
                input=page.get_by_label('GCP-001 A X',exact=True);input.fill('211.123456789');input.press('Tab')
                page.wait_for_function('!dirty && !saving')
                check('numeric_edit_not_rounded_on_save',abs(page.evaluate('doc.points[0].a[0]')-211.123456789)<1e-12)
                coordinates=page.evaluate('doc.points.map(p=>({a:p.a,b:p.b}))')
                page.locator('#mode').select_option('test');page.wait_for_function('mode==="test" && !loading')
                check('preview_switch_keeps_canonical_coordinates',page.evaluate('doc.points.map(p=>({a:p.a,b:p.b}))')==coordinates)
                page.locator('#mode').select_option('native');page.wait_for_function('mode==="native" && !loading')
                page.locator('#annotator').fill('QA_SYNTHETIC_CLICKS_NOT_HUMAN_GT');page.locator('#annotator').press('Tab')
                page.locator('#status').select_option('completed');page.wait_for_function('doc.status==="completed" && !dirty && !saving')
                page.locator('#reviewMode').check();check('review_locks_coordinate_inputs',page.get_by_label('GCP-001 A X',exact=True).is_disabled())
                page.locator('#reviewer').fill('QA_REVIEW');page.locator('#reviewer').press('Tab')
                page.get_by_label('GCP-001 审核已看',exact=True).check();page.get_by_label('GCP-002 审核已看',exact=True).check();page.locator('#approve').click()
                page.wait_for_function('doc.review.status==="approved" && !dirty && !saving')
                check('independent_review_persists',page.evaluate('doc.review.status')=='approved')
                revision=page.evaluate('doc.revision');page.reload();page.wait_for_function('doc && !loading')
                check('saved_points_and_review_restore',page.evaluate('doc.revision')==revision and page.evaluate('doc.review.status')=='approved')
                page.locator('#next').click();page.wait_for_function('(id)=>doc.pair_id!==id && !loading',arg=pair)
                page.locator('#prev').click();page.wait_for_function('(id)=>doc.pair_id===id && !loading',arg=pair)
                check('pair_navigation_preserves_saved_points',page.evaluate('doc.points.length')==2)
                exported=page.request.post(url+'/api/export').json();check('official_export_completed_points_only',exported['pairs']==1 and exported['points']==2)
                check('CSV_and_JSONL_download',page.request.get(url+exported['csv']).ok and page.request.get(url+exported['jsonl']).ok)
                page.screenshot(path=str(report_dir/'browser.png'),full_page=True)
                check('browser_has_no_javascript_errors',not errors)
                browser_version=browser.version
                browser.close()
            # Actually restart the server process and re-read persisted data.
            server.terminate();server.wait(timeout=15)
            server=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT);ready()
            with urllib.request.urlopen(url+'/api/pairs/'+pair) as response:
                restored=json.load(response)['annotation']
            check('server_process_restart_preserves_annotation',restored['revision']==revision and len(restored['points'])==2)
            evaluation=None
            if args.matches_root:
                metric_command=[str(args.eval_python),'-m','tools.gcp_annotator.evaluate','--data-root',str(args.data_root),
                                '--annotations',str(temporary/'labels/annotations'),'--matches-root',str(args.matches_root),
                                '--output',str(temporary/'evaluation')]
                subprocess.run(metric_command,cwd=REPO,check=True,capture_output=True,text=True)
                evaluation=json.loads((temporary/'evaluation/summary.json').read_text())
                check('actual_saved_model_NPZ_all_four_variants',len(evaluation['overall'])==4 and all(v['valid_pairs']==1 for v in evaluation['overall'].values()))
                check('exact_source_experiment_environment',not evaluation['source_version_mismatch'])
            check('original_manifest_images_unchanged',all(sha_file(path)==digest for path,digest in originals.items()))
            report=dict(created_at=now(),result='passed',checks=checks,pair_id=pair,
                        purpose='SOFTWARE QA ONLY: synthetic browser clicks are not human ground truth; temporary labels deleted',
                        coordinate_precision=result,original_hashes=originals,browser_version=browser_version,
                        source_sha256={name:sha_file(REPO/'tools/gcp_annotator'/name) for name in
                                       ['app.py','core.py','evaluate.py','static/app.js','static/geometry.js',
                                        'static/index.html','static/style.css','tests/browser_smoke.py']},
                        saved_match_evaluation=dict(versions=evaluation['versions'],variants=list(evaluation['overall']),
                                                    npz=[{k:v[k] for k in ['pair_id','npz','npz_sha256']} for v in evaluation['inputs']]) if evaluation else None,
                        screenshot=str(report_dir/'browser.png'))
            write_json(report_dir/'verification.json',report)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        finally:
            if server.poll() is None:server.terminate();server.wait(timeout=15)
            log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--matches-root',type=Path)
    p.add_argument('--eval-python',type=Path,default=Path(sys.executable))
    run(p.parse_args())


if __name__=='__main__':main()
